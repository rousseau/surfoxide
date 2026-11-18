//! Détection continue des collisions (Q2.0) : pour chaque sommet déplacé, la fraction `t*` du pas qu'il peut
//! parcourir avant de toucher la surface.
//!
//! Les sommets vont de leur ancienne position à la nouvelle en ligne droite, à vitesse constante, tous
//! ensemble. Un contact est :
//! - **sommet contre face** : le sommet est dans le plan de la face (coplanarité de quatre points) et dans le
//!   triangle ;
//! - **arête contre arête** : les quatre extrémités sont coplanaires et les arêtes se croisent.
//!
//! La coplanarité de quatre points mobiles est le zéro du polynôme cubique `det[(b−a)(t), (c−a)(t), (d−a)(t)]`
//! (Bridson et al. 2002) : ses racines de `[0, 1]` sont isolées par celles de sa dérivée, puis affinées par
//! bissection.
//!
//! Tolérances (millimètres, `L` = arête moyenne) : épaisseur de garde `η = 1e-6 · L` (les éléments à moins de `η`
//! sont comptés en contact), racines de la cubique à `1e-10` en `t`, cubique numériquement nulle quand ses
//! coefficients sont tous inférieurs à `1e-9 · L³` (quatre points coplanaires pendant tout le pas : recouvrement
//! 2D testé à 18 instants, `t* = 0` s'il y en a un). Limite : une racine double (contact tangent sans
//! traversée) n'est pas détectée ; le contrôle final de [`crate::safe_update_impact`] la rattrape.

use std::collections::{HashMap, HashSet};

use surfoxide_core::vec3::{self, Vec3};
use surfoxide_core::TriMesh;

/// Épaisseur de garde relative à l'arête moyenne.
pub const EPAISSEUR_REL: f64 = 1e-6;
/// Tolérance de la bissection sur `t`.
pub const TOLERANCE_T: f64 = 1e-10;
/// Seuil relatif (à `L³`) sous lequel la cubique de coplanarité est considérée comme nulle.
pub const NULLE_REL: f64 = 1e-9;
/// Nombre d'instants intérieurs testés quand la cubique est nulle (plus `t = 0` et `t = 1`).
pub const ECHANTILLONS_COPLANAIRES: usize = 16;
/// Marge de sécurité appliquée à `t*`.
pub const MARGE: f64 = 0.8;

fn det(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    vec3::dot(a, vec3::cross(b, c))
}

/// Coefficients `[c0, c1, c2, c3]` de `det[(b−a)(t), (c−a)(t), (d−a)(t)]` pour des points qui vont de `p0` à
/// `p1` (dans l'ordre `a, b, c, d`) en ligne droite, `t ∈ [0, 1]`.
pub fn cubique_coplanarite(p0: [Vec3; 4], p1: [Vec3; 4]) -> [f64; 4] {
    let u0 = vec3::sub(p0[1], p0[0]);
    let v0 = vec3::sub(p0[2], p0[0]);
    let w0 = vec3::sub(p0[3], p0[0]);
    let du = vec3::sub(vec3::sub(p1[1], p1[0]), u0);
    let dv = vec3::sub(vec3::sub(p1[2], p1[0]), v0);
    let dw = vec3::sub(vec3::sub(p1[3], p1[0]), w0);
    [
        det(u0, v0, w0),
        det(du, v0, w0) + det(u0, dv, w0) + det(u0, v0, dw),
        det(du, dv, w0) + det(du, v0, dw) + det(u0, dv, dw),
        det(du, dv, dw),
    ]
}

fn evalue(c: [f64; 4], t: f64) -> f64 {
    ((c[3] * t + c[2]) * t + c[1]) * t + c[0]
}

/// Racines réelles dans `]0, 1[` de la dérivée `c1 + 2 c2 t + 3 c3 t²`.
fn points_critiques(c: [f64; 4]) -> Vec<f64> {
    let (a, b, d) = (3.0 * c[3], 2.0 * c[2], c[1]);
    let echelle = a.abs().max(b.abs()).max(d.abs());
    let mut out = Vec::new();
    if echelle == 0.0 {
        return out;
    }
    if a.abs() <= 1e-14 * echelle {
        if b.abs() > 1e-14 * echelle {
            out.push(-d / b);
        }
    } else {
        let disc = b * b - 4.0 * a * d;
        if disc >= 0.0 {
            // Forme stable (évite la soustraction de deux nombres proches).
            let q = -0.5 * (b + 1.0_f64.copysign(b) * disc.sqrt());
            out.push(q / a);
            if q != 0.0 {
                out.push(d / q);
            }
        }
    }
    out.retain(|&t| t > 0.0 && t < 1.0);
    out.sort_by(f64::total_cmp);
    out
}

/// Racines de la cubique dans `[0, 1]`, en ordre croissant (sans les racines doubles : voir l'en-tête).
pub fn racines_cubique(c: [f64; 4]) -> Vec<f64> {
    let mut bornes = vec![0.0];
    bornes.extend(points_critiques(c));
    bornes.push(1.0);
    let mut racines = Vec::new();
    for w in bornes.windows(2) {
        let (mut lo, mut hi) = (w[0], w[1]);
        let (flo, fhi) = (evalue(c, lo), evalue(c, hi));
        if flo == 0.0 {
            racines.push(lo);
            continue;
        }
        if fhi == 0.0 {
            // La racine en `hi` est celle de l'intervalle suivant, sauf pour le dernier.
            if hi == 1.0 {
                racines.push(1.0);
            }
            continue;
        }
        if (flo < 0.0) == (fhi < 0.0) {
            continue;
        }
        while hi - lo > TOLERANCE_T {
            let m = 0.5 * (lo + hi);
            let fm = evalue(c, m);
            if fm == 0.0 {
                lo = m;
                hi = m;
                break;
            }
            if (fm < 0.0) == (flo < 0.0) {
                lo = m;
            } else {
                hi = m;
            }
        }
        racines.push(0.5 * (lo + hi));
    }
    racines
}

fn position(p0: Vec3, p1: Vec3, t: f64) -> Vec3 {
    vec3::lerp(p0, p1, t)
}

/// Coordonnées barycentriques de `x` (projeté dans le plan) dans le triangle `a, b, c`.
fn barycentriques(x: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<[f64; 3]> {
    let (v0, v1, v2) = (vec3::sub(b, a), vec3::sub(c, a), vec3::sub(x, a));
    let (d00, d01, d11) = (vec3::dot(v0, v0), vec3::dot(v0, v1), vec3::dot(v1, v1));
    let (d20, d21) = (vec3::dot(v2, v0), vec3::dot(v2, v1));
    let den = d00 * d11 - d01 * d01;
    if den.abs() <= 1e-30 {
        return None;
    }
    let v = (d11 * d20 - d01 * d21) / den;
    let w = (d00 * d21 - d01 * d20) / den;
    Some([1.0 - v - w, v, w])
}

/// Le sommet `x` est-il dans le triangle (à `tol` près en coordonnées barycentriques) ?
fn sommet_dans_triangle(x: Vec3, a: Vec3, b: Vec3, c: Vec3, tol: f64) -> bool {
    barycentriques(x, a, b, c).is_some_and(|l| l.iter().all(|&v| v >= -tol && v <= 1.0 + tol))
}

/// Les segments `p1p2` et `q1q2` (coplanaires à cet instant) se croisent-ils, à `tol` (paramètres) et `eta`
/// (mm) près ?
fn aretes_se_croisent(p1: Vec3, p2: Vec3, q1: Vec3, q2: Vec3, tol: f64, eta: f64) -> bool {
    let d1 = vec3::sub(p2, p1);
    let d2 = vec3::sub(q2, q1);
    let r = vec3::sub(p1, q1);
    let (a, e, b) = (vec3::dot(d1, d1), vec3::dot(d2, d2), vec3::dot(d1, d2));
    let (c, f) = (vec3::dot(d1, r), vec3::dot(d2, r));
    let den = a * e - b * b;
    if den > 1e-12 * a * e {
        let s = (b * f - c * e) / den;
        let u = (a * f - b * c) / den;
        let dans = |v: f64| v >= -tol && v <= 1.0 + tol;
        return dans(s) && dans(u);
    }
    // Arêtes parallèles : sur une même droite (à `eta` près) et avec recouvrement des projections.
    if a <= 0.0 {
        return false;
    }
    let dist_droite = vec3::norm(vec3::cross(d1, vec3::sub(q1, p1))) / a.sqrt();
    if dist_droite > eta {
        return false;
    }
    let (s1, s2) = (
        vec3::dot(d1, vec3::sub(q1, p1)) / a,
        vec3::dot(d1, vec3::sub(q2, p1)) / a,
    );
    s1.max(s2) >= -tol && s1.min(s2) <= 1.0 + tol
}

/// Élément en mouvement : quatre points (sommet + triangle, ou deux arêtes), de `p0` à `p1`.
#[derive(Clone, Copy)]
enum Paire {
    SommetFace,
    AreteArete,
}

fn dedans(paire: Paire, p0: [Vec3; 4], p1: [Vec3; 4], t: f64, tol: f64, eta: f64) -> bool {
    let p: Vec<Vec3> = (0..4).map(|i| position(p0[i], p1[i], t)).collect();
    match paire {
        Paire::SommetFace => sommet_dans_triangle(p[0], p[1], p[2], p[3], tol),
        Paire::AreteArete => aretes_se_croisent(p[0], p[1], p[2], p[3], tol, eta),
    }
}

/// Premier instant de `[0, 1]` où les quatre points (sommet + triangle `p[1..4]`, ou arête `p[0..2]` + arête
/// `p[2..4]`) se touchent, ou `None`. `echelle` est l'arête moyenne `L`.
fn premier_contact(paire: Paire, p0: [Vec3; 4], p1: [Vec3; 4], echelle: f64) -> Option<f64> {
    let eta = EPAISSEUR_REL * echelle;
    let tol = EPAISSEUR_REL;
    let c = cubique_coplanarite(p0, p1);
    let maximum = c.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if maximum <= NULLE_REL * echelle.powi(3) {
        // Quatre points coplanaires pendant tout le pas : recouvrement 2D à des instants échantillonnés.
        let n = ECHANTILLONS_COPLANAIRES + 1;
        let recouvre = (0..=n).any(|k| dedans(paire, p0, p1, k as f64 / n as f64, tol, eta));
        return recouvre.then_some(0.0);
    }
    racines_cubique(c)
        .into_iter()
        .find(|&t| dedans(paire, p0, p1, t, tol, eta))
}

/// Premier instant où le sommet `x` (de `x0` à `x1`) touche le triangle `f` (de `f0` à `f1`).
pub fn temps_sommet_face(x: (Vec3, Vec3), f: ([Vec3; 3], [Vec3; 3]), echelle: f64) -> Option<f64> {
    let p0 = [x.0, f.0[0], f.0[1], f.0[2]];
    let p1 = [x.1, f.1[0], f.1[1], f.1[2]];
    premier_contact(Paire::SommetFace, p0, p1, echelle)
}

/// Premier instant où l'arête `e` touche l'arête `g` (chacune : extrémités de départ, d'arrivée).
pub fn temps_arete_arete(e: ([Vec3; 2], [Vec3; 2]), g: ([Vec3; 2], [Vec3; 2]), echelle: f64) -> Option<f64> {
    let p0 = [e.0[0], e.0[1], g.0[0], g.0[1]];
    let p1 = [e.1[0], e.1[1], g.1[0], g.1[1]];
    premier_contact(Paire::AreteArete, p0, p1, echelle)
}

type Boite = (Vec3, Vec3);

fn boite_points(pts: &[Vec3], marge: f64) -> Boite {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in pts {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo.map(|v| v - marge), hi.map(|v| v + marge))
}

fn boites_se_touchent(a: &Boite, b: &Boite) -> bool {
    (0..3).all(|k| a.0[k] <= b.1[k] && b.0[k] <= a.1[k])
}

/// Grille uniforme de boîtes (sert de phase large).
struct Grille {
    cote: f64,
    cases: HashMap<[i64; 3], Vec<u32>>,
}

impl Grille {
    fn new(boites: &[Boite], cote: f64) -> Self {
        let mut cases: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        for (i, b) in boites.iter().enumerate() {
            let (lo, hi) = Self::plage(b, cote);
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        cases.entry([x, y, z]).or_default().push(i as u32);
                    }
                }
            }
        }
        Self { cote, cases }
    }

    fn plage(b: &Boite, cote: f64) -> ([i64; 3], [i64; 3]) {
        let f = |v: f64| (v / cote).floor() as i64;
        ([f(b.0[0]), f(b.0[1]), f(b.0[2])], [f(b.1[0]), f(b.1[1]), f(b.1[2])])
    }

    /// Indices des boîtes qui partagent une case avec `b` (avec doublons).
    fn candidats(&self, b: &Boite, out: &mut Vec<u32>) {
        out.clear();
        let (lo, hi) = Self::plage(b, self.cote);
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    if let Some(v) = self.cases.get(&[x, y, z]) {
                        out.extend_from_slice(v);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }
}

/// Pour chaque sommet de `mesh`, fraction `t* ∈ [0, 1]` du pas `ancienne position → new` possible avant le
/// premier contact (sommet-face ou arête-arête) avec la surface elle-même ou avec `obstacle` (fixe) ; `None` s'il
/// n'y a aucun contact pendant le pas (un contact exactement à `t = 1` compte : `Some(1.0)`). Les sommets de
/// `mesh` qui ne bougent pas valent `None`.
pub fn fractions_avant_contact(mesh: &TriMesh, new: &[Vec3], obstacle: Option<&TriMesh>) -> Vec<Option<f64>> {
    assert_eq!(new.len(), mesh.n_vertices());
    let n_mesh = mesh.n_vertices();
    let mut p0: Vec<Vec3> = mesh.vertices().to_vec();
    let mut p1: Vec<Vec3> = new.to_vec();
    let mut faces: Vec<[u32; 3]> = mesh.faces().to_vec();
    if let Some(o) = obstacle {
        p0.extend_from_slice(o.vertices());
        p1.extend_from_slice(o.vertices());
        let decalage = n_mesh as u32;
        faces.extend(o.faces().iter().map(|f| f.map(|v| v + decalage)));
    }
    let mut tstar: Vec<Option<f64>> = vec![None; n_mesh];
    let bouge: Vec<bool> = p0.iter().zip(&p1).map(|(a, b)| a != b).collect();
    if !bouge.iter().any(|&b| b) {
        return tstar;
    }
    // Échelle : arête moyenne de `mesh`.
    let mut somme = 0.0;
    let mut n_aretes = 0usize;
    let mut aretes: Vec<[u32; 2]> = Vec::new();
    {
        let mut vues: HashSet<(u32, u32)> = HashSet::new();
        for f in &faces {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                let cle = (a.min(b), a.max(b));
                if vues.insert(cle) {
                    aretes.push([cle.0, cle.1]);
                    if (cle.0 as usize) < n_mesh {
                        somme += vec3::dist(p0[cle.0 as usize], p0[cle.1 as usize]);
                        n_aretes += 1;
                    }
                }
            }
        }
    }
    if n_aretes == 0 {
        return tstar;
    }
    let echelle = somme / n_aretes as f64;
    let eta = EPAISSEUR_REL * echelle;

    let boite_sommet = |v: usize| boite_points(&[p0[v], p1[v]], eta);
    let boites_sommets: Vec<Boite> = (0..p0.len()).map(boite_sommet).collect();
    let boite_union = |ids: &[u32]| -> Boite {
        let pts: Vec<Vec3> = ids.iter().flat_map(|&v| [p0[v as usize], p1[v as usize]]).collect();
        boite_points(&pts, eta)
    };
    let boites_faces: Vec<Boite> = faces.iter().map(|f| boite_union(f)).collect();
    let boites_aretes: Vec<Boite> = aretes.iter().map(|e| boite_union(e)).collect();
    // Case : un peu plus que l'arête moyenne plus le déplacement moyen (les boîtes balayées sont petites devant elle).
    let etendue_moyenne = boites_faces
        .iter()
        .map(|b| (0..3).map(|k| b.1[k] - b.0[k]).fold(0.0_f64, f64::max))
        .sum::<f64>()
        / boites_faces.len() as f64;
    let cote = (1.5 * etendue_moyenne).max(echelle).max(1e-9);
    let grille_sommets = Grille::new(&boites_sommets, cote);
    let grille_faces = Grille::new(&boites_faces, cote);
    let grille_aretes = Grille::new(&boites_aretes, cote);

    let mut maj = |ids: &[u32], t: f64| {
        for &v in ids {
            if (v as usize) < n_mesh && bouge[v as usize] {
                tstar[v as usize] = Some(tstar[v as usize].map_or(t, |u| u.min(t)));
            }
        }
    };
    let mut cand: Vec<u32> = Vec::new();

    // Sommet contre face, côté faces mobiles : toutes les faces dont un sommet bouge contre tous les sommets.
    for (fi, f) in faces.iter().enumerate() {
        if !f.iter().any(|&v| bouge[v as usize]) {
            continue;
        }
        grille_sommets.candidats(&boites_faces[fi], &mut cand);
        for &v in &cand {
            if f.contains(&v) || !boites_se_touchent(&boites_sommets[v as usize], &boites_faces[fi]) {
                continue;
            }
            let x = (p0[v as usize], p1[v as usize]);
            let tri0 = f.map(|i| p0[i as usize]);
            let tri1 = f.map(|i| p1[i as usize]);
            if let Some(t) = temps_sommet_face(x, (tri0, tri1), echelle) {
                maj(&[v, f[0], f[1], f[2]], t);
            }
        }
    }
    // Sommet mobile contre faces immobiles (les faces mobiles ont été vues ci-dessus).
    for v in 0..n_mesh {
        if !bouge[v] {
            continue;
        }
        grille_faces.candidats(&boites_sommets[v], &mut cand);
        for &fi in &cand {
            let f = faces[fi as usize];
            if f.iter().any(|&i| bouge[i as usize]) || f.contains(&(v as u32)) {
                continue;
            }
            if !boites_se_touchent(&boites_sommets[v], &boites_faces[fi as usize]) {
                continue;
            }
            let tri0 = f.map(|i| p0[i as usize]);
            if let Some(t) = temps_sommet_face((p0[v], p1[v]), (tri0, tri0), echelle) {
                maj(&[v as u32], t);
            }
        }
    }
    // Arête contre arête.
    for (ei, e) in aretes.iter().enumerate() {
        let e_bouge = e.iter().any(|&v| bouge[v as usize]);
        if !e_bouge {
            continue;
        }
        grille_aretes.candidats(&boites_aretes[ei], &mut cand);
        for &gi in &cand {
            let g = aretes[gi as usize];
            if gi as usize == ei || e.iter().any(|v| g.contains(v)) {
                continue;
            }
            let g_bouge = g.iter().any(|&v| bouge[v as usize]);
            if g_bouge && gi < ei as u32 {
                continue; // paire vue depuis l'autre arête
            }
            if !boites_se_touchent(&boites_aretes[ei], &boites_aretes[gi as usize]) {
                continue;
            }
            let ea = (e.map(|i| p0[i as usize]), e.map(|i| p1[i as usize]));
            let ga = (g.map(|i| p0[i as usize]), g.map(|i| p1[i as usize]));
            if let Some(t) = temps_arete_arete(ea, ga, echelle) {
                maj(&[e[0], e[1], g[0], g[1]], t);
            }
        }
    }
    tstar
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_check::self_intersections;
    use surfoxide_core::shapes::icosphere;

    const TRI: [Vec3; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

    fn proche(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-8
    }

    /// Racines d'une cubique par la formule trigonométrique (oracle indépendant de la bissection).
    fn racines_trigonometriques(c: [f64; 4]) -> Vec<f64> {
        let (a, b, cc, d) = (c[3], c[2], c[1], c[0]);
        let (p, q) = (
            (3.0 * a * cc - b * b) / (3.0 * a * a),
            (2.0 * b.powi(3) - 9.0 * a * b * cc + 27.0 * a * a * d) / (27.0 * a.powi(3)),
        );
        let decalage = -b / (3.0 * a);
        assert!(p < 0.0, "trois racines réelles attendues");
        let m = 2.0 * (-p / 3.0).sqrt();
        let arg = (3.0 * q / (p * m)).clamp(-1.0, 1.0);
        let theta = arg.acos() / 3.0;
        let mut r: Vec<f64> = (0..3)
            .map(|k| m * (theta - 2.0 * std::f64::consts::PI * k as f64 / 3.0).cos() + decalage)
            .filter(|t| (0.0..=1.0).contains(t))
            .collect();
        r.sort_by(f64::total_cmp);
        r
    }

    #[test]
    fn racines_d_une_cubique_connue() {
        // (t − 0,2)(t − 0,5)(t − 0,9) = t³ − 1,6 t² + 0,73 t − 0,09
        let r = racines_cubique([-0.09, 0.73, -1.6, 1.0]);
        assert_eq!(r.len(), 3);
        for (a, b) in r.iter().zip([0.2, 0.5, 0.9]) {
            assert!(proche(*a, b), "{r:?}");
        }
        // Polynôme constant non nul : aucune racine.
        assert!(racines_cubique([1.0, 0.0, 0.0, 0.0]).is_empty());
        // Racines hors de [0, 1] ignorées : (t − 0,5)(t − 2)(t + 1) = t³ − 1,5 t² − 1,5 t + 1
        let r = racines_cubique([1.0, -1.5, -1.5, 1.0]);
        assert_eq!(r.len(), 1, "{r:?}");
        assert!(proche(r[0], 0.5), "{r:?}");
    }

    #[test]
    fn racines_contre_la_formule_trigonometrique() {
        // Cubiques à trois racines réelles distinctes tirées pseudo-aléatoirement (graine fixée).
        let mut etat = 12345u64;
        let mut u = || {
            etat = etat.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((etat >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut verifies = 0;
        for _ in 0..500 {
            let mut r = [u() * 3.0 - 1.0, u() * 3.0 - 1.0, u() * 3.0 - 1.0];
            r.sort_by(f64::total_cmp);
            if r[1] - r[0] < 0.05 || r[2] - r[1] < 0.05 {
                continue;
            }
            let k = 0.5 + u();
            let c = [
                -k * r[0] * r[1] * r[2],
                k * (r[0] * r[1] + r[0] * r[2] + r[1] * r[2]),
                -k * (r[0] + r[1] + r[2]),
                k,
            ];
            let a = racines_cubique(c);
            let b = racines_trigonometriques(c);
            assert_eq!(a.len(), b.len(), "{c:?} : {a:?} / {b:?}");
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() < 1e-7, "{c:?} : {a:?} / {b:?}");
            }
            verifies += 1;
        }
        assert!(verifies > 100);
    }

    #[test]
    fn collision_frontale_sommet_contre_face() {
        // Le sommet (0,25 ; 0,25 ; 1) tombe sur le triangle z = 0 jusqu'en z = −1 : contact en z = 0, t = 1/2.
        let t = temps_sommet_face(([0.25, 0.25, 1.0], [0.25, 0.25, -1.0]), (TRI, TRI), 1.0).unwrap();
        assert!(proche(t, 0.5), "{t}");
        // Triangle qui monte de z = 0 à z = 1 contre un sommet immobile en z = 0,25 : t = 1/4.
        let haut = TRI.map(|p| [p[0], p[1], 1.0]);
        let t = temps_sommet_face(([0.25, 0.25, 0.25], [0.25, 0.25, 0.25]), (TRI, haut), 1.0).unwrap();
        assert!(proche(t, 0.25), "{t}");
    }

    #[test]
    fn collision_rasante_sommet_contre_face() {
        // Presque parallèle au plan : de (0,1 ; 0,1 ; 0,01) à (0,5 ; 0,1 ; −0,01), traverse z = 0 à t = 1/2, x = 0,3.
        let t = temps_sommet_face(([0.1, 0.1, 0.01], [0.5, 0.1, -0.01]), (TRI, TRI), 1.0).unwrap();
        assert!(proche(t, 0.5), "{t}");
        // Même pente, mais la traversée du plan se fait hors du triangle (x = 1,1 à t = 1/2).
        assert!(temps_sommet_face(([0.1, 0.1, 0.01], [2.1, 0.1, -0.01]), (TRI, TRI), 1.0).is_none());
        // Glisse au-dessus du triangle sans le traverser.
        assert!(temps_sommet_face(([0.1, 0.1, 0.01], [0.5, 0.1, 0.01]), (TRI, TRI), 1.0).is_none());
        // Traverse le plan à t = 1/2 en x = 0,9 − 0,0 mais en dehors par y : (0,9 ; 0,3) est hors du triangle (x + y > 1).
        assert!(temps_sommet_face(([0.9, 0.3, 0.01], [0.9, 0.3, -0.01]), (TRI, TRI), 1.0).is_none());
    }

    #[test]
    fn collision_arete_contre_arete() {
        let e = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let g = [[0.0, -1.0, 1.0], [0.0, 1.0, 1.0]];
        // g descend de z = 1 à z = −1 : croise e (z = 0) en t = 1/2.
        let g1 = g.map(|p| [p[0], p[1], p[2] - 2.0]);
        let t = temps_arete_arete((e, e), (g, g1), 1.0).unwrap();
        assert!(proche(t, 0.5), "{t}");
        // e monte aussi de z = 0 à z = 1 : z_g(t) = 1 − 2t, z_e(t) = t, contact en t = 1/3.
        let e1 = e.map(|p| [p[0], p[1], p[2] + 1.0]);
        let t = temps_arete_arete((e, e1), (g, g1), 1.0).unwrap();
        assert!(proche(t, 1.0 / 3.0), "{t}");
        // Décalée de 2 en x : les droites se croisent en projection mais pas les segments.
        let gd = g.map(|p| [p[0] + 2.0, p[1], p[2]]);
        let gd1 = g1.map(|p| [p[0] + 2.0, p[1], p[2]]);
        assert!(temps_arete_arete((e, e), (gd, gd1), 1.0).is_none());
    }

    #[test]
    fn cas_coplanaire_pendant_tout_le_pas() {
        // Dans le plan du triangle : entre dedans (t* = 0), ou s'en éloigne (aucun contact).
        let t = temps_sommet_face(([2.0, 2.0, 0.0], [0.2, 0.2, 0.0]), (TRI, TRI), 1.0).unwrap();
        assert_eq!(t, 0.0);
        assert!(temps_sommet_face(([2.0, 2.0, 0.0], [3.0, 3.0, 0.0]), (TRI, TRI), 1.0).is_none());
        // Arêtes coplanaires parallèles, sur la même droite, qui se chevauchent à l'arrivée.
        let e = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let g0 = [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let g1 = [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]];
        assert_eq!(temps_arete_arete((e, e), (g0, g1), 1.0), Some(0.0));
        let g2 = [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
        assert!(temps_arete_arete((e, e), (g0, g2), 1.0).is_none());
    }

    #[test]
    fn fractions_sur_une_sphere() {
        let s = icosphere(3, 10.0);
        let mut new = s.vertices().to_vec();
        // Le sommet 0 traverse la sphère de part en part ; le sommet 5 s'éloigne un peu.
        // (décalé de l'axe : pas de coplanarité par symétrie avec l'antipode)
        new[0] = vec3::add(vec3::scale(new[0], -1.3), [1.0, 0.7, 0.3]);
        new[5] = vec3::scale(new[5], 1.01);
        let t = fractions_avant_contact(&s, &new, None);
        let t0 = t[0].expect("le sommet 0 traverse la sphère");
        // À la main : de r = 10 à r ≈ −13 le long d'un diamètre, la face opposée (r = −10) est touchée à 20/23 ≈ 0,87.
        assert!((t0 - 20.0 / 23.0).abs() < 0.01, "{t0}");
        assert_eq!(t[5], None);
        // Sans déplacement : rien.
        assert!(fractions_avant_contact(&s, s.vertices(), None)
            .iter()
            .all(|v| v.is_none()));
    }

    #[test]
    fn mise_a_jour_au_temps_d_impact_garde_la_surface_saine() {
        let mut s = icosphere(3, 10.0);
        let mut b = s.clone();
        let mut new = s.vertices().to_vec();
        new[0] = vec3::scale(new[0], -1.3);
        new[5] = vec3::scale(new[5], 1.02);
        let depart = s.vertices()[0];
        let _ = crate::safe_update(&mut b, &new);
        let _ = crate::safe_update_impact(&mut s, &new, None);
        assert!(self_intersections(&s).is_empty());
        assert_eq!(s.vertices()[5], new[5]);
        // Le défaut ramène le sommet 0 en place ; la variante le fait avancer, mais pas jusqu'à la cible.
        assert_eq!(b.vertices()[0], depart);
        assert!(vec3::dist(depart, s.vertices()[0]) > 1.0);
        assert!(vec3::dist(depart, s.vertices()[0]) < vec3::dist(depart, new[0]));
    }

    /// Sphère plissée : rayon modulé par des harmoniques.
    fn sphere_plissee() -> TriMesh {
        let mut s = icosphere(2, 10.0);
        for p in s.vertices_mut() {
            let n = vec3::normalize(*p);
            let r = 10.0 + 0.8 * (4.0 * n[0]).sin() * (3.0 * n[1]).cos() + 0.5 * (5.0 * n[2]).sin();
            *p = vec3::scale(n, r);
        }
        s
    }

    #[test]
    fn dix_mille_pas_aleatoires_sans_intersection() {
        let mut s = sphere_plissee();
        assert!(self_intersections(&s).is_empty());
        let mut etat = 2026u64;
        let mut u = || {
            etat = etat.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((etat >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let (mut avec_contact, mut deplacement) = (0usize, 0.0);
        for pas in 0..10_000 {
            let mut new = s.vertices().to_vec();
            // Un sous-ensemble de sommets bouge, parfois de plusieurs arêtes (≈ 1 mm) : des contacts sont fréquents.
            let amplitude = if pas % 2 == 0 { 0.6 } else { 3.0 };
            for (i, p) in new.iter_mut().enumerate() {
                if (i + pas) % 7 == 0 {
                    *p = vec3::add(*p, [amplitude * u(), amplitude * u(), amplitude * u()]);
                }
            }
            let t = fractions_avant_contact(&s, &new, None);
            if t.iter().any(|v| v.is_some()) {
                avec_contact += 1;
            }
            let avant = s.vertices().to_vec();
            crate::safe_update_impact(&mut s, &new, None);
            assert!(self_intersections(&s).is_empty(), "intersection au pas {pas}");
            deplacement += avant
                .iter()
                .zip(s.vertices())
                .map(|(a, b)| vec3::dist(*a, *b))
                .sum::<f64>();
        }
        assert!(avec_contact > 1_000, "contacts détectés : {avec_contact}");
        assert!(deplacement > 0.0);
    }

    #[test]
    fn la_variante_perd_moins_de_deplacement_que_le_defaut() {
        let base = sphere_plissee();
        let mut etat = 7u64;
        let mut u = || {
            etat = etat.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((etat >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let (mut defaut, mut variante) = (base.clone(), base);
        let (mut sd, mut sv) = (crate::StepStats::default(), crate::StepStats::default());
        for _ in 0..30 {
            let mut new = defaut.vertices().to_vec();
            for p in new.iter_mut() {
                *p = vec3::add(*p, [1.5 * u(), 1.5 * u(), 1.5 * u()]);
            }
            crate::safe_step(&mut defaut, &new, crate::SafeUpdateMode::Binary, &mut sd);
            let mut new = variante.vertices().to_vec();
            for p in new.iter_mut() {
                *p = vec3::add(*p, [1.5 * u(), 1.5 * u(), 1.5 * u()]);
            }
            crate::safe_step(&mut variante, &new, crate::SafeUpdateMode::Impact, &mut sv);
        }
        assert!(sd.lost_fraction() > 0.0 && sv.lost_fraction() > 0.0);
        assert!(sv.held_tenth < sd.held_tenth, "{sd:?} / {sv:?}");
    }

    /// Empreinte exacte (bits) des sommets d'une surface.
    fn empreinte(m: &TriMesh) -> Vec<[u64; 3]> {
        m.vertices().iter().map(|p| p.map(f64::to_bits)).collect()
    }

    #[test]
    fn le_defaut_binaire_reste_identique_bit_a_bit() {
        use surfoxide_core::shapes::torus;
        for base in [sphere_plissee(), torus(24, 12, 8.0, 3.0)] {
            let (mut a, mut b) = (base.clone(), base);
            let mut stats = crate::StepStats::default();
            let mut etat = 99u64;
            let mut u = || {
                etat = etat.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((etat >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
            };
            let mut held = 0;
            for _ in 0..40 {
                let new: Vec<Vec3> = a
                    .vertices()
                    .iter()
                    .map(|p| vec3::add(*p, [1.2 * u(), 1.2 * u(), 1.2 * u()]))
                    .collect();
                held += crate::safe_update(&mut a, &new);
                crate::safe_step(&mut b, &new, crate::SafeUpdateMode::Binary, &mut stats);
            }
            assert_eq!(empreinte(&a), empreinte(&b));
            assert_eq!(stats.held, held);
            assert!(held > 0, "le test doit exercer des retours");
        }
    }
}
