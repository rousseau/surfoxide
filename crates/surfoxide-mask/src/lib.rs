//! `surfoxide-mask` : masques d'hémisphère à partir des segmentations BOUNTI-19 et multi-BOUNTI-43.
//!
//! Chaque label est classé en un tissu ([`Tissue`]) et un côté ([`Side`]). Le masque « blanc » d'un
//! hémisphère, dont le bord est la surface WM/GM, réunit la substance blanche, le ventricule latéral et
//! les noyaux gris centraux du côté (comme fetpype `surface_processing`). Les structures médianes non
//! latéralisées (cavum, 3ᵉ ventricule, corps calleux et « intracerebral background » de multi-BOUNTI)
//! sont partagées entre les deux côtés : chaque voxel va au côté le plus proche, par propagation depuis
//! les voxels latéralisés. Cela ferme la face médiale du masque. On garde ensuite la plus grande
//! composante 26-connexe et on remplit ses cavités.
//!
//! [`segmentation_qc`] contrôle la segmentation elle-même : la substance blanche n'y touche jamais le fond et
//! le cortex est bordé de LCR, sauf erreur (tissu étiqueté fond par la segmentation).

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::str::FromStr;

use surfoxide_core::components::{fill_cavities, largest_component};
use surfoxide_core::{Connectivity, Hemisphere, Volume};

/// Classe de tissu d'un label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tissue {
    /// LCR extra-cérébral.
    Csf,
    /// Cortex.
    CorticalGm,
    /// Intérieur de la surface blanche : substance blanche, ventricules latéraux, noyaux gris centraux.
    White,
    /// Hors des hémisphères cérébraux : tronc, cervelet, 4ᵉ ventricule.
    Excluded,
}

/// Côté d'un label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
    /// Structure médiane, partagée entre les deux hémisphères.
    Midline,
    /// Sans objet (structures exclues).
    None,
}

/// Table de labels d'une segmentation.
#[derive(Debug, Clone)]
pub struct LabelTable {
    pub name: &'static str,
    entries: HashMap<i32, (Tissue, Side)>,
}

impl LabelTable {
    fn new(name: &'static str, rows: &[(i32, Tissue, Side)]) -> Self {
        Self {
            name,
            entries: rows.iter().map(|&(l, t, s)| (l, (t, s))).collect(),
        }
    }

    /// BOUNTI-19 (table validée : `CRL-Cinema/configs/bounti19_to_cinema9.json`).
    pub fn bounti19() -> Self {
        use Side::*;
        use Tissue::*;
        Self::new(
            "bounti19",
            &[
                (1, Csf, Left),
                (2, Csf, Right),
                (3, CorticalGm, Left),
                (4, CorticalGm, Right),
                (5, White, Left),
                (6, White, Right),
                (7, White, Left),     // ventricule latéral
                (8, White, Right),    // ventricule latéral
                (9, White, Midline),  // cavum septum pellucidum
                (10, Excluded, None), // tronc cérébral
                (11, Excluded, None), // cervelet
                (12, Excluded, None),
                (13, Excluded, None), // vermis
                (14, White, Left),    // noyaux gris centraux
                (15, White, Right),
                (16, White, Left), // thalamus
                (17, White, Right),
                (18, White, Midline), // 3ᵉ ventricule
                (19, Excluded, None), // 4ᵉ ventricule
            ],
        )
    }

    /// Multi-BOUNTI-43 (Uus et al. 2026 ; table de `fetal-folding-geometry/src/parcellate.py`).
    /// Les labels impairs de 1 à 24 sont à droite, les pairs à gauche.
    pub fn multibounti43() -> Self {
        use Side::*;
        use Tissue::*;
        let mut rows = Vec::new();
        for l in 1..=24 {
            let side = if l % 2 == 1 { Right } else { Left };
            let tissue = if l <= 12 { CorticalGm } else { White };
            rows.push((l, tissue, side));
        }
        rows.extend([
            (25, White, Midline), // corps calleux
            (26, White, Midline), // « intracerebral background », surtout de la WM profonde
            (27, Excluded, None), // tronc cérébral
            (28, Excluded, None), // cervelet
            (29, Excluded, None),
            (30, Excluded, None), // vermis
            (31, White, Right),   // noyau caudé
            (32, White, Left),
            (33, White, Right), // noyau lenticulaire
            (34, White, Left),
            (35, White, Right), // thalamus
            (36, White, Left),
            (37, Csf, Right),
            (38, Csf, Left),
            (39, White, Right), // ventricule latéral
            (40, White, Left),
            (41, White, Midline), // cavum septum pellucidum
            (42, White, Midline), // 3ᵉ ventricule
            (43, Excluded, None), // 4ᵉ ventricule
        ]);
        Self::new("multibounti43", &rows)
    }

    /// Tissu et côté d'un label ; `None` pour le fond et les labels inconnus.
    pub fn get(&self, label: i32) -> Option<(Tissue, Side)> {
        self.entries.get(&label).copied()
    }
}

impl FromStr for LabelTable {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "bounti19" | "bounti" => Ok(Self::bounti19()),
            "multibounti43" | "multibounti" | "multi-bounti" => Ok(Self::multibounti43()),
            _ => Err(format!("table de labels inconnue « {s} » (bounti19 ou multibounti43)")),
        }
    }
}

impl fmt::Display for LabelTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name)
    }
}

/// Côté de chaque voxel : 0 aucun, 1 gauche, 2 droit. Les voxels médians reçoivent le côté du voxel
/// latéralisé le plus proche, par propagation (26-connexe) à travers les voxels médians seulement.
pub fn assign_sides(labels: &Volume<i32>, table: &LabelTable) -> Volume<u8> {
    let mut side = vec![0u8; labels.len()];
    let mut midline = vec![false; labels.len()];
    let mut queue = VecDeque::new();
    for (idx, &l) in labels.data().iter().enumerate() {
        match table.get(l).map(|(_, s)| s) {
            Some(Side::Left) => side[idx] = 1,
            Some(Side::Right) => side[idx] = 2,
            Some(Side::Midline) => midline[idx] = true,
            _ => {}
        }
    }
    // Sources : voxels latéralisés voisins d'un voxel médian.
    let offsets = Connectivity::TwentySix.offsets();
    let neighbors = |idx: usize| {
        let [i, j, k] = labels.coords(idx);
        offsets.iter().filter_map(move |o| {
            let (a, b, c) = (i as isize + o[0], j as isize + o[1], k as isize + o[2]);
            labels
                .contains(a, b, c)
                .then(|| labels.index(a as usize, b as usize, c as usize))
        })
    };
    for (idx, &s) in side.iter().enumerate() {
        if s != 0 && neighbors(idx).any(|n| midline[n]) {
            queue.push_back(idx);
        }
    }
    while let Some(idx) = queue.pop_front() {
        let s = side[idx];
        for n in neighbors(idx).collect::<Vec<_>>() {
            if midline[n] && side[n] == 0 {
                side[n] = s;
                queue.push_back(n);
            }
        }
    }
    labels.with_data(side)
}

/// Voxels d'un tissu donné du côté `hemi` (côtés déjà attribués par [`assign_sides`]).
pub fn tissue_mask(
    labels: &Volume<i32>,
    sides: &Volume<u8>,
    table: &LabelTable,
    tissue: Tissue,
    hemi: Hemisphere,
) -> Volume<bool> {
    let want = if hemi.is_left() { 1 } else { 2 };
    labels.with_data(
        labels
            .data()
            .iter()
            .zip(sides.data())
            .map(|(&l, &s)| s == want && table.get(l).is_some_and(|(t, _)| t == tissue))
            .collect(),
    )
}

/// Masque blanc fermé d'un hémisphère : tissu [`Tissue::White`] du côté, plus grande composante
/// 26-connexe, cavités (au sens 6-connexe du fond) remplies.
pub fn white_mask(labels: &Volume<i32>, table: &LabelTable, hemi: Hemisphere) -> Volume<bool> {
    hemisphere_masks(labels, table, hemi).0
}

/// Masque blanc fermé (comme [`white_mask`]) et masque du cortex d'un hémisphère.
pub fn hemisphere_masks(labels: &Volume<i32>, table: &LabelTable, hemi: Hemisphere) -> (Volume<bool>, Volume<bool>) {
    let sides = assign_sides(labels, table);
    let raw = tissue_mask(labels, &sides, table, Tissue::White, hemi);
    let white = fill_cavities(&largest_component(&raw, Connectivity::TwentySix), Connectivity::Six);
    let cortex = tissue_mask(labels, &sides, table, Tissue::CorticalGm, hemi);
    (white, cortex)
}

/// Aire de contact substance blanche–fond (mm², un hémisphère) au-delà de laquelle la segmentation est
/// suspecte. Réglé sur la cohorte CRL2025 : aucun hémisphère BOUNTI-19 au-dessus (au plus 30 mm²).
pub const WHITE_BACKGROUND_SUSPECT: f64 = 100.0;

/// Part de la frontière externe du cortex au contact direct du fond au-delà de laquelle la segmentation est
/// suspecte. Réglé sur la cohorte CRL2025 : au plus 0,12 pour BOUNTI-19 (médiane 0,012).
pub const CORTEX_BACKGROUND_SUSPECT: f64 = 0.15;

/// Contrôle d'une segmentation pour un hémisphère, en faces de voxels.
///
/// Dans une segmentation correcte, la substance blanche est entourée de cortex, de ventricules ou de noyaux,
/// et le cortex est bordé de LCR extra-cérébral. Du tissu étiqueté fond (label 0) par erreur se voit de deux
/// façons :
/// - la substance blanche latéralisée touche le fond (BOUNTI-19 sur la cohorte CRL2025 : médiane 3 mm², au
///   plus 30 mm² par hémisphère ; multi-BOUNTI-43 : jusqu'à 1 500 mm², corrélation de 0,93 avec le volume
///   manquant) ;
/// - le cortex perd sa couche externe et touche le fond au lieu du LCR : la part de sa frontière externe au
///   contact du fond, au plus 0,12 pour BOUNTI-19, monte de 0,17 à 0,43 dans les sessions multi-BOUNTI-43 qui
///   ont perdu du cortex sans toucher à la substance blanche.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegQc {
    /// Contact substance blanche latéralisée–fond (mm²), sans les structures médianes.
    pub white_background_mm2: f64,
    /// Contact cortex–fond (mm²).
    pub cortex_background_mm2: f64,
    /// Contact cortex–LCR extra-cérébral (mm²).
    pub cortex_csf_mm2: f64,
}

impl SegQc {
    /// Part de la frontière externe du cortex (fond ou LCR) au contact du fond.
    pub fn cortex_background_fraction(&self) -> f64 {
        let outer = self.cortex_background_mm2 + self.cortex_csf_mm2;
        if outer > 0.0 {
            self.cortex_background_mm2 / outer
        } else {
            0.0
        }
    }

    /// Vrai si l'un des deux critères dépasse son seuil ([`WHITE_BACKGROUND_SUSPECT`],
    /// [`CORTEX_BACKGROUND_SUSPECT`]).
    pub fn suspect(&self) -> bool {
        self.white_background_mm2 > WHITE_BACKGROUND_SUSPECT
            || self.cortex_background_fraction() > CORTEX_BACKGROUND_SUSPECT
    }
}

/// Contrôle de la segmentation d'un hémisphère (voir [`SegQc`]).
pub fn segmentation_qc(labels: &Volume<i32>, table: &LabelTable, hemi: Hemisphere) -> SegQc {
    let side = if hemi.is_left() { Side::Left } else { Side::Right };
    // 0 fond, 1 substance blanche du côté, 2 cortex du côté, 3 LCR, 4 autre.
    let class = |l: i32| match (l, table.get(l)) {
        (0, _) => 0u8,
        (_, Some((Tissue::White, s))) if s == side => 1,
        (_, Some((Tissue::CorticalGm, s))) if s == side => 2,
        (_, Some((Tissue::Csf, _))) => 3,
        _ => 4,
    };
    let [nx, ny, nz] = labels.dims();
    let s = labels.spacing();
    // Aire d'une face de voxel perpendiculaire à chaque axe.
    let face = [s[1] * s[2], s[0] * s[2], s[0] * s[1]];
    let mut qc = SegQc {
        white_background_mm2: 0.0,
        cortex_background_mm2: 0.0,
        cortex_csf_mm2: 0.0,
    };
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let here = class(labels.get(i, j, k));
                for (a, (di, dj, dk)) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)].into_iter().enumerate() {
                    let (ii, jj, kk) = (i + di, j + dj, k + dk);
                    if ii >= nx || jj >= ny || kk >= nz {
                        continue;
                    }
                    let there = class(labels.get(ii, jj, kk));
                    let pair = (here.min(there), here.max(there));
                    match pair {
                        (0, 1) => qc.white_background_mm2 += face[a],
                        (0, 2) => qc.cortex_background_mm2 += face[a],
                        (2, 3) => qc.cortex_csf_mm2 += face[a],
                        _ => {}
                    }
                }
            }
        }
    }
    qc
}

/// Sommets corticaux d'une surface : vrai si le voxel de cortex le plus proche est à moins de `max_dist`
/// mm. Les autres forment la face médiale non corticale (ventricules, noyaux gris, plan médian), où
/// l'épaisseur n'a pas de sens. La carte est nettoyée par `smooth` passes de vote majoritaire sur le
/// premier anneau.
pub fn cortical_vertices(
    mesh: &surfoxide_core::TriMesh,
    cortex: &Volume<bool>,
    max_dist: f64,
    smooth: usize,
) -> Vec<bool> {
    let d2 = surfoxide_core::distance::edt_squared(cortex);
    let dims = cortex.dims();
    let mut label: Vec<bool> = mesh
        .vertices()
        .iter()
        .map(|&p| {
            let Some(q) = cortex.world_to_voxel(p) else {
                return false;
            };
            let idx: [usize; 3] = std::array::from_fn(|a| (q[a].round().max(0.0) as usize).min(dims[a] - 1));
            d2.get(idx[0], idx[1], idx[2]) <= max_dist * max_dist
        })
        .collect();
    let nb = surfoxide_core::adjacency::vertex_neighbors(mesh);
    for _ in 0..smooth {
        label = (0..label.len())
            .map(|i| {
                let yes = nb[i].iter().filter(|&&j| label[j as usize]).count() + label[i] as usize;
                2 * yes > nb[i].len() + 1
            })
            .collect();
    }
    label
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::volume::IDENTITY;

    /// Petite segmentation synthétique : WM gauche (5) et droite (6) séparées par un plan de cavum (9),
    /// un ventricule gauche (7) enclavé dans la WM gauche, un voxel de cervelet (11).
    fn exemple() -> Volume<i32> {
        let dims = [11, 6, 6];
        let mut v = Volume::filled(dims, 0, IDENTITY);
        for k in 1..5 {
            for j in 1..5 {
                for i in 1..10 {
                    let l = match i {
                        1..=4 => 5,
                        5 => 9,
                        _ => 6,
                    };
                    v.set(i, j, k, l);
                }
            }
        }
        v.set(2, 2, 2, 7);
        v.set(8, 2, 0, 11);
        v
    }

    #[test]
    fn plan_median_partage() {
        let labels = exemple();
        let sides = assign_sides(&labels, &LabelTable::bounti19());
        // Le cavum est à égale distance des deux côtés : il est attribué en entier, et de façon
        // déterministe, à l'un des deux.
        let s = sides.get(5, 2, 2);
        assert!(s == 1 || s == 2);
        assert_eq!(sides.get(2, 2, 2), 1);
        assert_eq!(sides.get(8, 2, 0), 0);
    }

    #[test]
    fn masque_blanc_gauche() {
        let labels = exemple();
        let m = white_mask(&labels, &LabelTable::bounti19(), Hemisphere::Left);
        assert!(m.get(2, 2, 2), "le ventricule gauche fait partie du masque");
        assert!(m.get(1, 1, 1));
        assert!(!m.get(7, 2, 2), "la WM droite n'en fait pas partie");
        assert!(!m.get(8, 2, 0), "le cervelet n'en fait pas partie");
    }

    #[test]
    fn sommets_corticaux_d_une_demi_coquille() {
        use surfoxide_core::shapes::icosphere;
        // Sphère de rayon 5 ; cortex seulement au-dessus du plan z = 0, entre 5 et 7 mm.
        let n = 41;
        let c = (n as f64 - 1.0) / 2.0 * 0.5;
        let mut aff = IDENTITY;
        for (a, row) in aff.iter_mut().take(3).enumerate() {
            row[a] = 0.5;
            row[3] = -c;
        }
        let mut cortex = Volume::filled([n, n, n], false, aff);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = cortex.voxel_to_world([i as f64, j as f64, k as f64]);
                    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                    cortex.set(i, j, k, p[2] > 0.0 && (5.0..7.0).contains(&r));
                }
            }
        }
        let s = icosphere(3, 5.0);
        let lab = cortical_vertices(&s, &cortex, 1.0, 2);
        for (p, &l) in s.vertices().iter().zip(&lab) {
            if p[2] > 1.5 {
                assert!(l, "sommet {p:?} devrait être cortical");
            }
            if p[2] < -1.5 {
                assert!(!l, "sommet {p:?} ne devrait pas être cortical");
            }
        }
    }

    #[test]
    fn tables_completes() {
        let b = LabelTable::bounti19();
        assert!((1..=19).all(|l| b.get(l).is_some()));
        let m = LabelTable::multibounti43();
        assert!((1..=43).all(|l| m.get(l).is_some()));
        assert_eq!(m.get(14), Some((Tissue::White, Side::Left)));
        assert_eq!(m.get(1), Some((Tissue::CorticalGm, Side::Right)));
    }

    /// Sur une vraie segmentation : masques gauche et droit disjoints, d'un seul tenant, du bon côté.
    #[test]
    #[ignore = "données réelles : bash scripts/fetch-data.sh"]
    fn masques_reels() {
        let p = surfoxide_io::data::segmentation("sub-0dfxc", "ses-01", "bounti19");
        let (labels, _) = surfoxide_io::nifti::read_labels(p).unwrap();
        let table = LabelTable::bounti19();
        let l = white_mask(&labels, &table, Hemisphere::Left);
        let r = white_mask(&labels, &table, Hemisphere::Right);
        assert!(!l.data().iter().zip(r.data()).any(|(&a, &b)| a && b));
        let centroid_x = |m: &Volume<bool>| {
            let (mut s, mut n) = (0.0, 0.0);
            for (idx, _) in m.data().iter().enumerate().filter(|(_, &b)| b) {
                let [i, j, k] = m.coords(idx);
                s += m.voxel_to_world([i as f64, j as f64, k as f64])[0];
                n += 1.0;
            }
            s / n
        };
        assert!(centroid_x(&l) < centroid_x(&r));
        let vol = |m: &Volume<bool>| m.data().iter().filter(|&&b| b).count() as f64 * 0.125;
        assert!(vol(&l) > 10_000.0 && vol(&r) > 10_000.0, "{} {}", vol(&l), vol(&r));
    }

    #[test]
    fn controle_segmentation_substance_blanche_et_cortex() {
        // Boule de substance blanche gauche (5) dans du cortex (3), lui-même dans du LCR (1) : aucun contact
        // avec le fond. On retire ensuite tout d'un côté : la substance blanche et le cortex touchent le fond.
        let t = LabelTable::bounti19();
        let mut v = Volume::filled([14, 14, 14], 0i32, surfoxide_core::volume::IDENTITY);
        let inside = |a: usize, lo: usize, hi: usize| a >= lo && a <= hi;
        for k in 0..14 {
            for j in 0..14 {
                for i in 0..14 {
                    let cube = |lo, hi| inside(i, lo, hi) && inside(j, lo, hi) && inside(k, lo, hi);
                    if cube(5, 8) {
                        v.set(i, j, k, 5);
                    } else if cube(3, 10) {
                        v.set(i, j, k, 3);
                    } else if cube(2, 11) {
                        v.set(i, j, k, 1);
                    }
                }
            }
        }
        let qc = segmentation_qc(&v, &t, Hemisphere::Left);
        assert_eq!(qc.white_background_mm2, 0.0);
        assert_eq!(qc.cortex_background_mm2, 0.0);
        // Faces externes du cube de cortex 8 × 8 × 8.
        assert_eq!(qc.cortex_csf_mm2, 6.0 * 64.0);
        assert!(!qc.suspect());
        assert_eq!(segmentation_qc(&v, &t, Hemisphere::Right).cortex_csf_mm2, 0.0);
        for k in 0..14 {
            for j in 0..14 {
                for i in 9..14 {
                    v.set(i, j, k, 0);
                }
            }
        }
        let qc = segmentation_qc(&v, &t, Hemisphere::Left);
        // Face x = 8 | 9 du cube 4 × 4 de substance blanche : 16 faces de 1 mm².
        assert_eq!(qc.white_background_mm2, 16.0);
        // Cortex au contact du fond : face x = 8 | 9 hors substance blanche (64 − 16).
        assert_eq!(qc.cortex_background_mm2, 48.0);
        assert!(qc.suspect());
    }
}
