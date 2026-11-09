//! `surfoxide-topofix` : correction topologique voxelique (Bazin & Pham 2005, 2007).
//!
//! L'objet est la région `φ < 0` d'une fonction de niveaux (en général la distance signée d'un masque).
//! On le reconstruit par croissance contrainte : un voxel n'est ajouté que s'il est *simple*, c'est-à-dire
//! si son ajout ne change pas la topologie (au sens de la connexité (26, 6) : objet 26-connexe, fond
//! 6-connexe). Deux variantes, comme chez Bazin :
//!
//! - croissance de l'objet depuis le minimum de φ, par φ croissant, jusqu'au niveau 0 : les anses sont
//!   coupées ;
//! - croissance du fond depuis le bord du volume, par φ décroissant, jusqu'au niveau 0 : les tunnels et
//!   les cavités sont bouchés.
//!
//! Chaque variante donne un objet homéomorphe à une boule (surface de genre 0). On garde par défaut celle
//! qui modifie le moins de voxels. La stratégie [`Strategy::PerDefect`] fait ce choix défaut par défaut : les
//! voxels où les deux corrections diffèrent sont regroupés en composantes 26-connexes (les défauts), chacune
//! prend le côté qui change le moins de voxels, puis une croissance contrainte garantit le genre 0. La fonction de niveaux est ensuite corrigée là où le signe change, pour
//! que le marching cubes à connexité cohérente (crate `surfoxide-extract`) produise exactement cette topologie
//! en gardant la précision sous-voxel ailleurs.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use surfoxide_core::Volume;

pub mod simple;

pub use simple::is_simple;

/// Variante de la correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Croissance de l'objet : coupe les anses.
    GrowObject,
    /// Croissance du fond : bouche les tunnels.
    GrowBackground,
    /// Les deux, en gardant celle qui modifie le moins de voxels.
    Best,
    /// Les deux, avec le choix du côté qui change le moins de voxels pour chaque défaut séparément.
    PerDefect,
}

/// Côté retenu pour un défaut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Les voxels du défaut qui appartiennent à l'objet initial sont retirés (anse coupée).
    Cut,
    /// Les voxels du défaut qui n'appartiennent pas à l'objet initial sont ajoutés (tunnel bouché).
    Plug,
}

/// Un défaut : composante 26-connexe des voxels où les deux corrections diffèrent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Defect {
    /// Nombre de voxels de la composante.
    pub size: usize,
    /// Voxels retirés à l'objet initial si on coupe (côté `Cut`).
    pub cost_cut: usize,
    /// Voxels ajoutés à l'objet initial si on bouche (côté `Plug`).
    pub cost_plug: usize,
    /// Côté choisi : le moins coûteux, la coupe en cas d'égalité.
    pub side: Side,
}

/// Résultat d'une correction.
#[derive(Debug, Clone)]
pub struct Correction {
    /// Objet corrigé, homéomorphe à une boule.
    pub object: Volume<bool>,
    /// Fonction de niveaux corrigée : négative exactement sur `object`.
    pub levelset: Volume<f64>,
    /// Voxels ajoutés à l'objet initial `φ < 0`.
    pub added: usize,
    /// Voxels retirés de l'objet initial.
    pub removed: usize,
    /// Variante retenue.
    pub strategy: Strategy,
    /// Voxels modifiés par la croissance de l'objet seule et par la croissance du fond seule (`Best` et
    /// `PerDefect` calculent les deux).
    pub single_costs: Option<(usize, usize)>,
    /// Défauts (stratégie `PerDefect` seulement).
    pub defects: Vec<Defect>,
    /// Voxels choisis par les défauts que la croissance contrainte finale n'a pas pu poser sans changer la
    /// topologie (stratégie `PerDefect` seulement).
    pub unrecovered: usize,
}

/// Erreur de correction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopoFixError {
    /// Aucun voxel avec φ < 0.
    EmptyObject,
    /// L'objet touche le bord du volume : il faut le recadrer avec une marge.
    TouchesBorder,
}

impl std::fmt::Display for TopoFixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TopoFixError::EmptyObject => write!(f, "objet vide (aucun voxel avec φ < 0)"),
            TopoFixError::TouchesBorder => write!(
                f,
                "l'objet touche le bord du volume : recadrer avec une marge d'au moins un voxel"
            ),
        }
    }
}

impl std::error::Error for TopoFixError {}

/// Entrée du tas : priorité (déjà orientée pour un tas max), puis indice pour départager.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Item(f64, usize);

impl Eq for Item {}

impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0).then_with(|| other.1.cmp(&self.1))
    }
}

impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Voisinage 3×3×3 de `idx` dans `set`, sous forme de 27 bits (bit `(di+1) + 3(dj+1) + 9(dk+1)`).
/// Les voxels hors du volume sont considérés comme hors de l'ensemble.
fn neighborhood(set: &[bool], dims: [usize; 3], idx: usize) -> u32 {
    let i = idx % dims[0];
    let j = (idx / dims[0]) % dims[1];
    let k = idx / (dims[0] * dims[1]);
    let mut bits = 0u32;
    let mut b = 0;
    for dk in -1..=1isize {
        for dj in -1..=1isize {
            for di in -1..=1isize {
                let (a, c, e) = (i as isize + di, j as isize + dj, k as isize + dk);
                if a >= 0
                    && c >= 0
                    && e >= 0
                    && (a as usize) < dims[0]
                    && (c as usize) < dims[1]
                    && (e as usize) < dims[2]
                    && set[a as usize + dims[0] * (c as usize + dims[1] * e as usize)]
                {
                    bits |= 1 << b;
                }
                b += 1;
            }
        }
    }
    bits
}

fn neighbors26(dims: [usize; 3], idx: usize) -> impl Iterator<Item = usize> {
    let i = (idx % dims[0]) as isize;
    let j = ((idx / dims[0]) % dims[1]) as isize;
    let k = (idx / (dims[0] * dims[1])) as isize;
    (0..27).filter(|&b| b != 13).filter_map(move |b| {
        let (a, c, e) = (i + b % 3 - 1, j + (b / 3) % 3 - 1, k + b / 9 - 1);
        (a >= 0 && c >= 0 && e >= 0 && (a as usize) < dims[0] && (c as usize) < dims[1] && (e as usize) < dims[2])
            .then(|| a as usize + dims[0] * (c as usize + dims[1] * e as usize))
    })
}

fn on_border(dims: [usize; 3], idx: usize) -> bool {
    let i = idx % dims[0];
    let j = (idx / dims[0]) % dims[1];
    let k = idx / (dims[0] * dims[1]);
    i == 0 || j == 0 || k == 0 || i + 1 == dims[0] || j + 1 == dims[1] || k + 1 == dims[2]
}

/// Croissance de l'objet depuis le minimum de φ, par φ croissant, sur les voxels `φ < 0`.
fn grow_object(phi: &Volume<f64>) -> Vec<bool> {
    let dims = phi.dims();
    let p = phi.data();
    let seed = (0..p.len())
        .min_by(|&a, &b| p[a].total_cmp(&p[b]).then(a.cmp(&b)))
        .expect("volume non vide");
    let mut obj = vec![false; p.len()];
    let mut heap = BinaryHeap::new();
    heap.push(Item(-p[seed], seed));
    while let Some(Item(_, v)) = heap.pop() {
        if obj[v] {
            continue;
        }
        // Le premier voxel, isolé, n'est pas « simple » (il crée une composante) : on l'ajoute d'office.
        if v != seed && !is_simple(neighborhood(&obj, dims, v)) {
            continue; // sera reproposé si un voisin est ajouté
        }
        obj[v] = true;
        for n in neighbors26(dims, v) {
            if !obj[n] && p[n] < 0.0 {
                heap.push(Item(-p[n], n));
            }
        }
    }
    obj
}

/// Croissance du fond depuis le bord du volume, par φ décroissant, sur les voxels `φ ≥ 0`.
fn grow_background(phi: &Volume<f64>) -> Vec<bool> {
    let dims = phi.dims();
    let p = phi.data();
    // `obj` est le complément du fond : au départ, tout sauf la couche du bord.
    let mut obj: Vec<bool> = (0..p.len()).map(|i| !on_border(dims, i)).collect();
    let mut heap = BinaryHeap::new();
    for v in (0..p.len()).filter(|&i| !obj[i]) {
        for n in neighbors26(dims, v) {
            if obj[n] && p[n] >= 0.0 {
                heap.push(Item(p[n], n));
            }
        }
    }
    while let Some(Item(_, v)) = heap.pop() {
        if !obj[v] {
            continue;
        }
        // Retirer v de l'objet préserve la topologie si et seulement si v est simple pour l'objet.
        if !is_simple(neighborhood(&obj, dims, v)) {
            continue;
        }
        obj[v] = false;
        for n in neighbors26(dims, v) {
            if obj[n] && p[n] >= 0.0 {
                heap.push(Item(p[n], n));
            }
        }
    }
    obj
}

/// Composantes 26-connexes des voxels de `mask` ; renvoie pour chaque voxel l'indice de sa composante (ou
/// `usize::MAX`) et le nombre de composantes.
fn components26(mask: &[bool], dims: [usize; 3]) -> (Vec<usize>, usize) {
    let mut label = vec![usize::MAX; mask.len()];
    let mut n = 0;
    let mut stack = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || label[start] != usize::MAX {
            continue;
        }
        label[start] = n;
        stack.push(start);
        while let Some(v) = stack.pop() {
            for w in neighbors26(dims, v) {
                if mask[w] && label[w] == usize::MAX {
                    label[w] = n;
                    stack.push(w);
                }
            }
        }
        n += 1;
    }
    (label, n)
}

/// Croissance contrainte depuis `seed` (objet homéomorphe à une boule) : n'ajoute que des points simples,
/// pris parmi `allowed`, par φ croissant. La topologie de `seed` est donc conservée.
fn grow_within(seed: &[bool], allowed: &[bool], p: &[f64], dims: [usize; 3]) -> Vec<bool> {
    let mut obj = seed.to_vec();
    let mut heap = BinaryHeap::new();
    for v in (0..obj.len()).filter(|&v| obj[v]) {
        for n in neighbors26(dims, v) {
            if !obj[n] && allowed[n] {
                heap.push(Item(-p[n], n));
            }
        }
    }
    while let Some(Item(_, v)) = heap.pop() {
        if obj[v] || !is_simple(neighborhood(&obj, dims, v)) {
            continue; // reproposé si un voisin est ajouté
        }
        obj[v] = true;
        for n in neighbors26(dims, v) {
            if !obj[n] && allowed[n] {
                heap.push(Item(-p[n], n));
            }
        }
    }
    obj
}

/// Correction défaut par défaut : `inner` (croissance de l'objet) est inclus dans l'objet initial, lui-même
/// inclus dans `outer` (croissance du fond). Retourne l'objet corrigé, les défauts et le nombre de voxels
/// choisis non posés.
fn per_defect(inner: &[bool], outer: &[bool], p: &[f64], dims: [usize; 3]) -> (Vec<bool>, Vec<Defect>, usize) {
    let diff: Vec<bool> = inner.iter().zip(outer).map(|(&a, &b)| b && !a).collect();
    let (label, n) = components26(&diff, dims);
    let mut defects: Vec<Defect> = (0..n)
        .map(|_| Defect {
            size: 0,
            cost_cut: 0,
            cost_plug: 0,
            side: Side::Cut,
        })
        .collect();
    for v in (0..diff.len()).filter(|&v| diff[v]) {
        let d = &mut defects[label[v]];
        d.size += 1;
        if p[v] < 0.0 {
            d.cost_cut += 1;
        } else {
            d.cost_plug += 1;
        }
    }
    for d in &mut defects {
        d.side = if d.cost_plug < d.cost_cut {
            Side::Plug
        } else {
            Side::Cut
        };
    }
    // Objet recombiné : l'intérieur, plus les défauts bouchés.
    let wanted: Vec<bool> = (0..inner.len())
        .map(|v| inner[v] || (diff[v] && defects[label[v]].side == Side::Plug))
        .collect();
    // Dernière croissance contrainte : depuis l'intérieur (une boule), on ne pose que des points simples,
    // donc le résultat est une boule quelle que soit l'interaction entre défauts voisins.
    let object = grow_within(inner, &wanted, p, dims);
    let unrecovered = wanted.iter().zip(&object).filter(|(&w, &o)| w && !o).count();
    (object, defects, unrecovered)
}

/// Corrige la topologie de l'objet `φ < 0` de façon qu'il soit homéomorphe à une boule.
///
/// `delta` (mm) est la valeur absolue donnée à φ sur les voxels dont le signe change : petite devant la
/// taille du voxel, elle place la surface corrigée tout contre ces voxels.
pub fn correct_topology(phi: &Volume<f64>, strategy: Strategy, delta: f64) -> Result<Correction, TopoFixError> {
    let dims = phi.dims();
    let p = phi.data();
    if !p.iter().any(|&x| x < 0.0) {
        return Err(TopoFixError::EmptyObject);
    }
    if (0..p.len()).any(|i| on_border(dims, i) && p[i] < 0.0) {
        return Err(TopoFixError::TouchesBorder);
    }
    let count = |obj: &[bool]| {
        let added = obj.iter().zip(p).filter(|(&o, &x)| o && x >= 0.0).count();
        let removed = obj.iter().zip(p).filter(|(&o, &x)| !o && x < 0.0).count();
        (added, removed)
    };
    let mut single_costs = None;
    let mut defects = Vec::new();
    let mut unrecovered = 0;
    let (object, used) = match strategy {
        Strategy::GrowObject => (grow_object(phi), Strategy::GrowObject),
        Strategy::GrowBackground => (grow_background(phi), Strategy::GrowBackground),
        Strategy::Best => {
            let a = grow_object(phi);
            let b = grow_background(phi);
            let (ca, cb) = (count(&a), count(&b));
            single_costs = Some((ca.0 + ca.1, cb.0 + cb.1));
            if cb.0 + cb.1 < ca.0 + ca.1 {
                (b, Strategy::GrowBackground)
            } else {
                (a, Strategy::GrowObject)
            }
        }
        Strategy::PerDefect => {
            let a = grow_object(phi);
            let b = grow_background(phi);
            let (ca, cb) = (count(&a), count(&b));
            single_costs = Some((ca.0 + ca.1, cb.0 + cb.1));
            let (o, d, u) = per_defect(&a, &b, p, dims);
            defects = d;
            unrecovered = u;
            (o, Strategy::PerDefect)
        }
    };
    let (added, removed) = count(&object);
    let levelset = object
        .iter()
        .zip(p)
        .map(|(&o, &x)| match (o, x < 0.0) {
            (true, true) | (false, false) => x,
            (true, false) => -delta,
            (false, true) => delta,
        })
        .collect();
    Ok(Correction {
        object: phi.with_data(object),
        levelset: phi.with_data(levelset),
        added,
        removed,
        strategy: used,
        single_costs,
        defects,
        unrecovered,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::distance::signed_distance;
    use surfoxide_core::volume::IDENTITY;

    /// Bilan topologique digital (26, 6) d'un objet : composantes de l'objet, composantes du fond, et
    /// caractéristique d'Euler du complexe cubique (union des cubes fermés).
    pub(crate) fn digital_topology(obj: &Volume<bool>) -> (usize, usize, i64) {
        use surfoxide_core::components::label_components;
        use surfoxide_core::Connectivity;
        let (_, fg) = label_components(obj, Connectivity::TwentySix);
        let (_, bg) = label_components(&obj.map(|&b| !b), Connectivity::Six);
        let mut cells = std::collections::HashSet::new();
        for (idx, _) in obj.data().iter().enumerate().filter(|(_, &b)| b) {
            let [i, j, k] = obj.coords(idx);
            for c in 0..3 {
                for b in 0..3 {
                    for a in 0..3 {
                        cells.insert((2 * i + a, 2 * j + b, 2 * k + c));
                    }
                }
            }
        }
        let chi: i64 = cells
            .iter()
            .map(|&(x, y, z)| if (x % 2 + y % 2 + z % 2) % 2 == 0 { 1 } else { -1 })
            .sum();
        (fg.len(), bg.len(), chi)
    }

    /// Tore plein discret (genre 1) dans un volume avec marge.
    fn solid_torus() -> Volume<bool> {
        let n = 32;
        let mut v = Volume::filled([n, n, 16], false, IDENTITY);
        for k in 0..16 {
            for j in 0..n {
                for i in 0..n {
                    let (x, y, z) = (i as f64 - 15.5, j as f64 - 15.5, k as f64 - 7.5);
                    let r = (x * x + y * y).sqrt() - 9.0;
                    v.set(i, j, k, r * r + z * z < 16.0);
                }
            }
        }
        v
    }

    #[test]
    fn tore_corrige_en_boule() {
        let torus = solid_torus();
        assert_eq!(digital_topology(&torus), (1, 1, 0), "le tore de test a χ = 0");
        let phi = signed_distance(&torus);
        for strategy in [Strategy::GrowObject, Strategy::GrowBackground, Strategy::Best] {
            let c = correct_topology(&phi, strategy, 0.01).unwrap();
            assert_eq!(digital_topology(&c.object), (1, 1, 1), "{strategy:?}");
            match c.strategy {
                Strategy::GrowObject => assert!(c.added == 0 && c.removed > 0),
                _ => assert!(c.removed == 0 && c.added > 0),
            }
            // La fonction de niveaux corrigée est négative exactement sur l'objet.
            assert!(c
                .object
                .data()
                .iter()
                .zip(c.levelset.data())
                .all(|(&o, &x)| o == (x < 0.0)));
        }
    }

    #[test]
    fn boule_inchangee() {
        let mut v = Volume::filled([20, 20, 20], false, IDENTITY);
        for k in 0..20 {
            for j in 0..20 {
                for i in 0..20 {
                    let d2 = (i as f64 - 9.5).powi(2) + (j as f64 - 9.5).powi(2) + (k as f64 - 9.5).powi(2);
                    v.set(i, j, k, d2 < 49.0);
                }
            }
        }
        let c = correct_topology(&signed_distance(&v), Strategy::Best, 0.01).unwrap();
        assert_eq!((c.added, c.removed), (0, 0));
        assert_eq!(c.object, v);
    }

    #[test]
    fn boule_creuse_remplie_ou_ouverte() {
        let mut v = Volume::filled([20, 20, 20], false, IDENTITY);
        for k in 0..20 {
            for j in 0..20 {
                for i in 0..20 {
                    let d2 = (i as f64 - 9.5).powi(2) + (j as f64 - 9.5).powi(2) + (k as f64 - 9.5).powi(2);
                    v.set(i, j, k, d2 < 64.0 && d2 > 9.0);
                }
            }
        }
        assert_eq!(digital_topology(&v).1, 2, "une cavité");
        let c = correct_topology(&signed_distance(&v), Strategy::Best, 0.01).unwrap();
        assert_eq!(digital_topology(&c.object), (1, 1, 1));
    }

    /// Plaque percée d'un petit trou traversant (un tunnel à boucher : 8 voxels) portant une anse fine (à
    /// couper : peu de voxels, alors que la boucher en coûterait des dizaines).
    fn plaque_percee_avec_anse() -> Volume<bool> {
        let mut v = Volume::filled([40, 40, 28], false, IDENTITY);
        for k in 8..16 {
            for j in 6..26 {
                for i in 6..26 {
                    v.set(i, j, k, true);
                }
            }
        }
        for k in 8..16 {
            v.set(15, 15, k, false);
        }
        // Anse : deux montants et une barre, au-dessus de la face supérieure (k = 15).
        for k in 16..21 {
            v.set(10, 9, k, true);
            v.set(10, 19, k, true);
        }
        for j in 9..20 {
            v.set(10, j, 20, true);
        }
        v
    }

    #[test]
    fn defaut_par_defaut_anse_coupee_et_tunnel_bouche() {
        let v = plaque_percee_avec_anse();
        let (fg, bg, chi) = digital_topology(&v);
        assert_eq!((fg, bg), (1, 1));
        assert_eq!(chi, -1, "deux tunnels (trou et anse) : χ = 1 − 2");
        let phi = signed_distance(&v);
        let best = correct_topology(&phi, Strategy::Best, 0.01).unwrap();
        let c = correct_topology(&phi, Strategy::PerDefect, 0.01).unwrap();
        assert_eq!(digital_topology(&c.object), (1, 1, 1));
        assert_eq!(c.unrecovered, 0);
        assert_eq!(c.defects.len(), 2, "un défaut par tunnel : {:?}", c.defects);
        // Le trou est bouché (1 voxel, contre 96 pour le couper) et l'anse coupée (1 voxel, contre 36 pour
        // combler son arche) : les deux corrections minimales.
        let trou = c
            .defects
            .iter()
            .find(|d| d.cost_cut > d.cost_plug)
            .expect("défaut du trou");
        assert_eq!((trou.side, trou.cost_plug), (Side::Plug, 1), "{trou:?}");
        let anse = c
            .defects
            .iter()
            .find(|d| d.cost_cut < d.cost_plug)
            .expect("défaut de l'anse");
        assert_eq!((anse.side, anse.cost_cut), (Side::Cut, 1), "{anse:?}");
        assert_eq!((c.added, c.removed), (1, 1));
        // Moins de voxels modifiés que la meilleure des deux corrections globales.
        assert!(
            c.added + c.removed < best.added + best.removed,
            "{} contre {}",
            c.added + c.removed,
            best.added + best.removed
        );
        let (a, b) = c.single_costs.unwrap();
        assert_eq!(best.added + best.removed, a.min(b));
        // Fonction de niveaux négative exactement sur l'objet.
        assert!(c
            .object
            .data()
            .iter()
            .zip(c.levelset.data())
            .all(|(&o, &x)| o == (x < 0.0)));
    }

    #[test]
    fn defaut_par_defaut_ne_depasse_jamais_la_meilleure_correction_globale() {
        // Objets aléatoires (bruit lissé seuillé) : genre 0 toujours, jamais plus de voxels modifiés que `Best`
        // (à une croissance finale près : `unrecovered` les compte).
        let mut seed = 0x2545F4914F6CDD1Du64;
        let n = 26;
        let mut testes = 0;
        for essai in 0..40 {
            let mut bruit = vec![0.0f64; n * n * n];
            for x in bruit.iter_mut() {
                *x = (seed_next(&mut seed) % 1000) as f64 / 1000.0;
            }
            // Lissage par moyenne 3×3×3 (deux passes) et seuil : blobs connexes avec anses et cavités.
            for _ in 0..2 {
                let copie = bruit.clone();
                for k in 1..n - 1 {
                    for j in 1..n - 1 {
                        for i in 1..n - 1 {
                            let mut s = 0.0;
                            for dk in 0..3 {
                                for dj in 0..3 {
                                    for di in 0..3 {
                                        s += copie[(i + di - 1) + n * ((j + dj - 1) + n * (k + dk - 1))];
                                    }
                                }
                            }
                            bruit[i + n * (j + n * k)] = s / 27.0;
                        }
                    }
                }
            }
            let seuil = 0.46 + 0.002 * essai as f64;
            let mut v = Volume::filled([n, n, n], false, IDENTITY);
            for k in 3..n - 3 {
                for j in 3..n - 3 {
                    for i in 3..n - 3 {
                        v.set(i, j, k, bruit[i + n * (j + n * k)] > seuil);
                    }
                }
            }
            if !v.data().iter().any(|&b| b) {
                continue;
            }
            let phi = signed_distance(&v);
            let best = correct_topology(&phi, Strategy::Best, 0.01).unwrap();
            let c = correct_topology(&phi, Strategy::PerDefect, 0.01).unwrap();
            assert_eq!(digital_topology(&c.object), (1, 1, 1), "essai {essai}");
            assert!(
                c.added + c.removed <= best.added + best.removed + c.unrecovered,
                "essai {essai} : {} contre {}",
                c.added + c.removed,
                best.added + best.removed
            );
            testes += 1;
        }
        assert!(testes >= 20, "trop peu d'objets testés ({testes})");
    }

    fn seed_next(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    /// Masque blanc réel : correction en quelques secondes, faible fraction de voxels modifiés.
    #[test]
    #[ignore = "données réelles : bash scripts/fetch-data.sh"]
    fn masque_reel() {
        use surfoxide_core::filter::{crop_to, gaussian_blur};
        for sub in ["sub-0dfxc/ses-01", "sub-0avbc/ses-02"] {
            let (s, e) = sub.split_once('/').unwrap();
            let p = surfoxide_io::data::segmentation(s, e, "bounti19");
            let (labels, _) = surfoxide_io::nifti::read_labels(p).unwrap();
            let table = surfoxide_mask::LabelTable::bounti19();
            for hemi in surfoxide_core::Hemisphere::BOTH {
                let t = std::time::Instant::now();
                let mask = surfoxide_mask::white_mask(&labels, &table, hemi);
                let (mask, _) = crop_to(&mask, |&b| b, 4, false).unwrap();
                let phi = gaussian_blur(&signed_distance(&mask), 0.25);
                let before = digital_topology(&phi.map(|&x| x < 0.0));
                let c = correct_topology(&phi, Strategy::Best, 0.01).unwrap();
                let after = digital_topology(&c.object);
                println!(
                    "{sub} {hemi} : avant {before:?}, après {after:?}, {:?}, +{} −{} voxels, {:.1} s",
                    c.strategy,
                    c.added,
                    c.removed,
                    t.elapsed().as_secs_f64()
                );
                assert_eq!(after, (1, 1, 1));
            }
        }
    }

    #[test]
    fn objet_au_bord_refuse() {
        let v = Volume::filled([4, 4, 4], -1.0, IDENTITY);
        assert_eq!(
            correct_topology(&v, Strategy::Best, 0.01).unwrap_err(),
            TopoFixError::TouchesBorder
        );
    }
}
