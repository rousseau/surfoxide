//! Fantômes B : couple blanche/piale analytique (plan qualité, Q0.2).
//!
//! Blanche : ellipsoïde (paramétré par une icosphère fine) creusé de sillons. Chaque sillon suit un arc de grand
//! cercle de plan Π (passant par l'origine, de normale `n`) : on enfonce d'abord radialement la surface d'un profil
//! lisse `D · (1 − (d/W)²)²`, `d` étant la distance au plan Π (mm) et `W` la demi-largeur du profil, atténué aux deux
//! bouts de l'arc ; on resserre ensuite les berges vers Π par `x ← x + (φ(s) − s) n`, `s = x · n`, avec
//! `φ(s) = s · m(|s|)` croissante (`m` vaut λ près de Π, monte à 1 entre 0,8 W et W). λ ne dépend que des
//! coordonnées orthogonales à `n` (profondeur sous l'ellipsoïde de la projection de x sur Π) : `λ = g / √(d² + g²)`,
//! `d` la demi-largeur du profil d'origine à cette profondeur, `2g` la largeur voulue ; les berges sont ainsi à
//! l'écart `2g` à toute profondeur où le profil d'origine est plus large, et le fond garde sa forme arrondie. Ce
//! resserrement ne change que la coordonnée le long de `n`, de façon strictement croissante à coordonnées
//! orthogonales fixées : c'est une bijection de l'espace, et la surface reste plongée. Pas de resserrement si la
//! largeur voulue atteint `2W`.
//!
//! Piale : blanche décalée le long de la normale de chaque sommet de `o`, la plus grande fonction de pente au plus
//! `SLOPE` (le long des arêtes) sous `min(T, max(ρ − ε, ρ/2))` : `T` l'épaisseur voulue, `ρ` le rayon de la plus grande boule
//! tangente extérieure en ce sommet qui ne contient aucun point de la blanche (boule « rétrécissante » : elle
//! tient compte à la fois de la courbure concave et de la berge d'en face d'un sillon, où ρ vaut le demi-écart).
//! Un point à moins de ρ le long de la normale a ce sommet pour plus proche point de la blanche. Borner la pente
//! évite que des décalages voisins très différents retournent des faces de la piale. Là où `o < T`, le sommet est
//! « au contact ».

use rayon::prelude::*;
use surfoxide_check::bvh::Bvh;
use surfoxide_check::distance::{closest_point_triangle, distance2_triangle};
use surfoxide_core::adjacency::vertex_normals;
use surfoxide_core::shapes::icosphere;
use surfoxide_core::vec3::{self, Vec3};
use surfoxide_core::TriMesh;

/// Un sillon.
#[derive(Debug, Clone, PartialEq)]
pub struct Sulcus {
    /// Centre de l'arc, direction depuis l'origine.
    pub center: Vec3,
    /// Direction de l'arc au centre (tangente à la sphère unité en `center`).
    pub direction: Vec3,
    /// Demi-longueur de l'arc (radians).
    pub half_length: f64,
    /// Profondeur (mm).
    pub depth: f64,
    /// Écart entre berges dans la partie resserrée (mm) ; `2 W` : pas de resserrement.
    pub width: f64,
}

/// Épaisseur voulue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Thickness {
    Constant(f64),
    /// `mid + amp · (3 u_z² − 1) / 2`, `u` direction du sommet sur l'icosphère : de `mid − amp/2` à `mid + amp`.
    Variable {
        mid: f64,
        amp: f64,
    },
}

impl Thickness {
    fn at(self, u: Vec3) -> f64 {
        match self {
            Thickness::Constant(t) => t,
            Thickness::Variable { mid, amp } => mid + amp * (3.0 * u[2] * u[2] - 1.0) / 2.0,
        }
    }
}

/// Paramètres d'un fantôme B.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalyticParams {
    /// Demi-axes de l'ellipsoïde (mm).
    pub axes: [f64; 3],
    /// Niveau de subdivision de l'icosphère (7 : 163 842 sommets).
    pub level: u32,
    /// Demi-largeur du profil des sillons (mm).
    pub profile_half_width: f64,
    pub sulci: Vec<Sulcus>,
    pub thickness: Thickness,
    /// Marge ε gardée avant le contact (mm).
    pub epsilon: f64,
}

/// Configurations figées.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Config {
    /// Sillons larges, épaisseur constante.
    Wide,
    /// Sillons étroits (écart 3,5 mm : plus étroits que les larges, piales encore séparées), épaisseur constante.
    Narrow,
    /// Sillons enfouis (écart 0,8 mm, moins que 2 T) : piales jointives à 2 ε près.
    Buried,
    /// Sillons larges, épaisseur variable.
    Variable,
}

/// Demi-axes par défaut : blanche de l'atlas dHCP à 28 SA (étendue 31,9 × 77,2 × 47,6 mm).
pub const AXES_28: [f64; 3] = [16.0, 38.5, 24.0];

fn unit(theta_deg: f64, phi_deg: f64) -> Vec3 {
    let (t, p) = (theta_deg.to_radians(), phi_deg.to_radians());
    [t.sin() * p.cos(), t.sin() * p.sin(), t.cos()]
}

/// Six arcs régulièrement répartis, tournés de `angle` degrés autour de leur centre.
fn six_sulci(depth: f64, width: f64, angle: f64) -> Vec<Sulcus> {
    [
        (60.0, 0.0),
        (60.0, 120.0),
        (60.0, 240.0),
        (120.0, 60.0),
        (120.0, 180.0),
        (120.0, 300.0),
    ]
    .iter()
    .enumerate()
    .map(|(k, &(t, p))| {
        let c = unit(t, p);
        // Repère tangent (e_θ, e_φ), direction tournée de angle (+ 60° un arc sur deux).
        let e_t = unit(t + 90.0, p);
        let e_p = vec3::cross(c, e_t);
        let a = (angle + if k % 2 == 0 { 0.0 } else { 60.0 }).to_radians();
        Sulcus {
            center: c,
            direction: vec3::add(vec3::scale(e_t, a.cos()), vec3::scale(e_p, a.sin())),
            half_length: 25f64.to_radians(),
            depth,
            width,
        }
    })
    .collect()
}

impl AnalyticParams {
    /// Paramètres d'une configuration figée ; `depth`, `width` et `angle` (degrés) remplacent ceux de la
    /// configuration s'ils sont donnés.
    pub fn config(c: Config, axes: [f64; 3], depth: Option<f64>, width: Option<f64>, angle: f64) -> Self {
        let w = 3.0;
        let (default_width, thickness) = match c {
            Config::Wide => (2.0 * w, Thickness::Constant(1.5)),
            Config::Narrow => (3.5, Thickness::Constant(1.5)),
            Config::Buried => (0.8, Thickness::Constant(1.5)),
            Config::Variable => (2.0 * w, Thickness::Variable { mid: 1.5, amp: 0.5 }),
        };
        Self {
            axes,
            level: 7,
            profile_half_width: w,
            sulci: six_sulci(depth.unwrap_or(4.0), width.unwrap_or(default_width), angle),
            thickness,
            epsilon: 0.05,
        }
    }
}

fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Atténuation aux bouts de l'arc et position du point par rapport à lui : 1 au milieu, 0 au-delà des bouts et
/// sur l'autre face.
/// `ramp` : part de la demi-longueur sur laquelle l'atténuation passe de 1 à 0, jusqu'au bout de l'arc.
fn taper(s: &Sulcus, n: Vec3, x: Vec3, ramp: f64) -> f64 {
    let q = vec3::sub(x, vec3::scale(n, vec3::dot(x, n)));
    let (a, b) = (vec3::dot(q, s.center), vec3::dot(q, s.direction));
    if a <= 0.0 {
        return 0.0;
    }
    let alpha = b.atan2(a).abs();
    let ramp = ramp * s.half_length;
    1.0 - smoothstep((alpha - (s.half_length - ramp)) / ramp)
}

/// Atténuations du creusement et du resserrement : le resserrement s'éteint plus tôt et plus progressivement, pour
/// ne pas cisailler les triangles au bout des sillons.
const CARVE_RAMP: f64 = 0.3;
const SQUEEZE_RAMP: f64 = 0.7;

/// Blanche analytique.
pub fn white_surface(p: &AnalyticParams) -> TriMesh {
    let sphere = icosphere(p.level, 1.0);
    let w = p.profile_half_width;
    let normals: Vec<Vec3> = p
        .sulci
        .iter()
        .map(|s| vec3::normalize(vec3::cross(s.center, s.direction)))
        .collect();
    let vertices: Vec<Vec3> = sphere
        .vertices()
        .par_iter()
        .map(|u| {
            let mut x = [u[0] * p.axes[0], u[1] * p.axes[1], u[2] * p.axes[2]];
            // Creusement radial.
            let r = vec3::norm(x);
            let mut dr = 0.0;
            for (s, &n) in p.sulci.iter().zip(&normals) {
                let d = vec3::dot(x, n).abs();
                if d < w {
                    let prof = (1.0 - (d / w).powi(2)).powi(2);
                    dr += s.depth * prof * taper(s, n, x, CARVE_RAMP);
                }
            }
            x = vec3::scale(x, 1.0 - dr / r);
            // Resserrement des berges : facteur λ fonction des seules coordonnées orthogonales à n (profondeur sous
            // l'ellipsoïde de la projection q de x sur Π, et atténuation), pour un écart g entre berges à toute
            // profondeur où le profil d'origine est plus large que g.
            for (s, &n) in p.sulci.iter().zip(&normals) {
                let sn = vec3::dot(x, n);
                let t = taper(s, n, x, SQUEEZE_RAMP);
                let carve = taper(s, n, x, CARVE_RAMP);
                if sn.abs() >= w || s.width >= 2.0 * w || t == 0.0 {
                    continue;
                }
                let q = vec3::sub(x, vec3::scale(n, sn));
                let rq = vec3::norm(q);
                let ellipsoid = 1.0 / (0..3).map(|k| (q[k] / rq / p.axes[k]).powi(2)).sum::<f64>().sqrt();
                let level = ((ellipsoid - rq) / (s.depth * carve)).clamp(0.0, 1.0);
                let half = w * (1.0 - level.sqrt()).sqrt();
                let g = s.width.max(1e-3) / 2.0;
                let lambda = (g / (half * half + g * g).sqrt()).min(1.0);
                let m = lambda + (1.0 - lambda) * smoothstep((sn.abs() / w - 0.8) / 0.2);
                x = vec3::add(x, vec3::scale(n, t * (sn * m - sn)));
            }
            x
        })
        .collect();
    TriMesh::new(vertices, sphere.faces().to_vec()).expect("connectivité de l'icosphère")
}

/// Rayon de la plus grande boule tangente en `w` du côté de `n` (unitaire) qui ne contient aucun point de `mesh`,
/// au plus `max` (boule rétrécissante : on remplace la boule par celle, tangente en `w`, qui passe par le plus proche
/// point de la surface de son centre, jusqu'à ce que ce point soit sur la sphère). Les faces incidentes au sommet
/// `i` sont ignorées : la normale du sommet n'étant pas celle de ses faces, leur plan coupe toute boule tangente en
/// `w` dès qu'il s'incline vers elle.
fn medial_radius(mesh: &TriMesh, bvh: &Bvh, i: u32, n: Vec3, max: f64) -> f64 {
    let v = mesh.vertices();
    let w = v[i as usize];
    let tri = |f: u32| mesh.faces()[f as usize].map(|i| v[i as usize]);
    let mut r = max;
    for _ in 0..100 {
        let c = vec3::add(w, vec3::scale(n, r));
        let Some((f, d2)) = bvh.nearest(c, |f| {
            if mesh.faces()[f as usize].contains(&i) {
                return f64::INFINITY;
            }
            let [a, b, cc] = tri(f);
            distance2_triangle(c, a, b, cc)
        }) else {
            return r;
        };
        if d2.sqrt() >= r * (1.0 - 1e-9) {
            return r;
        }
        let [a, b, cc] = tri(f);
        let q = vec3::sub(closest_point_triangle(c, a, b, cc), w);
        let h = vec3::dot(n, q);
        // Plus proche point en w lui-même (à l'arrondi près) : la boule est vide.
        if vec3::norm(q) <= 1e-9 * r.max(1.0) {
            return r;
        }
        if h <= 0.0 {
            return 0.0;
        }
        let next = vec3::dot(q, q) / (2.0 * h);
        if next >= r * (1.0 - 1e-12) {
            return r;
        }
        r = next;
    }
    r
}

/// Piale analytique de la blanche `white`, et décalage de chaque sommet ; `contact[i]` : décalage limité.
pub struct AnalyticPial {
    pub mesh: TriMesh,
    pub offset: Vec<f64>,
    pub target: Vec<f64>,
    pub contact: Vec<bool>,
}

/// Pente maximale du décalage de la piale le long des arêtes de la blanche.
pub const SLOPE: f64 = 0.5;

/// Plus grande fonction de pente au plus `slope` le long des arêtes sous `bound` (propagation de Dijkstra depuis
/// les plus petites valeurs).
fn lipschitz_envelope(mesh: &TriMesh, bound: &[f64], slope: f64) -> Vec<f64> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let nb = surfoxide_core::adjacency::vertex_neighbors(mesh);
    let v = mesh.vertices();
    let mut o = bound.to_vec();
    // Valeurs positives : leur représentation binaire est ordonnée comme elles.
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = o
        .iter()
        .enumerate()
        .map(|(i, x)| Reverse((x.max(0.0).to_bits(), i as u32)))
        .collect();
    while let Some(Reverse((bits, i))) = heap.pop() {
        let oi = f64::from_bits(bits);
        if oi > o[i as usize] {
            continue;
        }
        for &j in &nb[i as usize] {
            let c = oi + slope * vec3::dist(v[i as usize], v[j as usize]);
            if c < o[j as usize] {
                o[j as usize] = c;
                heap.push(Reverse((c.to_bits(), j)));
            }
        }
    }
    o
}

pub fn pial_surface(white: &TriMesh, p: &AnalyticParams) -> AnalyticPial {
    let n = vertex_normals(white);
    let bvh = Bvh::new(white);
    let sphere = icosphere(p.level, 1.0);
    let target: Vec<f64> = sphere.vertices().iter().map(|&u| p.thickness.at(u)).collect();
    let bound: Vec<f64> = (0..white.n_vertices())
        .into_par_iter()
        .map(|i| {
            let t = target[i];
            let rho = medial_radius(white, &bvh, i as u32, n[i], t + p.epsilon);
            t.min((rho - p.epsilon).max(rho / 2.0))
        })
        .collect();
    let offset = lipschitz_envelope(white, &bound, SLOPE);
    let contact = offset.iter().zip(&target).map(|(o, t)| o < t).collect();
    let vertices = white
        .vertices()
        .iter()
        .zip(&n)
        .zip(&offset)
        .map(|((w, n), o)| vec3::add(*w, vec3::scale(*n, *o)))
        .collect();
    AnalyticPial {
        mesh: TriMesh::new(vertices, white.faces().to_vec()).expect("même connectivité"),
        offset,
        target,
        contact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quatre_configurations_figees_conformes() {
        for c in [Config::Wide, Config::Narrow, Config::Buried, Config::Variable] {
            // Niveau 7, celui des configurations figées : la construction n'est pas garantie sur un maillage plus
            // grossier (au niveau 6, la piale des sillons étroits s'auto-intersecte).
            let p = AnalyticParams::config(c, AXES_28, None, None, 0.0);
            let w = white_surface(&p);
            let pial = pial_surface(&w, &p);
            let s = surfoxide_check::check_surface(&w);
            let q = surfoxide_check::check_surface(&pial.mesh);
            let pair = surfoxide_check::check_pair(&w, &pial.mesh, 0.02);
            assert!(s.valid && q.valid && pair.valid, "{c:?} : {s:?} {q:?} {pair:?}");
            // Hors contact, la piale est exactement à l'épaisseur voulue des sommets correspondants.
            for i in 0..w.n_vertices() {
                if !pial.contact[i] {
                    let d = vec3::dist(w.vertices()[i], pial.mesh.vertices()[i]);
                    assert!((d - pial.target[i]).abs() < 1e-9);
                }
            }
        }
    }
}
