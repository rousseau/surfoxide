//! Tests d'intersection exacts entre triangles, par les prédicats `orient2d` et `orient3d` de Shewchuk.
//!
//! Les triangles sont fermés : un contact (sommet sur une face, arêtes qui se touchent) compte comme une
//! intersection. Les cas des triangles qui partagent un sommet ou une arête sont traités à part, pour ne
//! pas signaler leur contact légitime.

use robust::{orient2d, orient3d, Coord, Coord3D};
use surfoxide_core::vec3::{self, Vec3};

fn c3(p: Vec3) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

/// Signe de `orient3d(a, b, c, d)` : −1, 0 ou 1.
fn o3(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> i8 {
    let v = orient3d(c3(a), c3(b), c3(c), c3(d));
    (v > 0.0) as i8 - (v < 0.0) as i8
}

/// Projection 2D sur le plan de coordonnées le plus proche du plan du triangle (abc).
fn projector(a: Vec3, b: Vec3, c: Vec3) -> impl Fn(Vec3) -> Coord<f64> {
    let n = vec3::triangle_normal(a, b, c);
    let axis = (0..3)
        .max_by(|&i, &j| n[i].abs().total_cmp(&n[j].abs()))
        .unwrap();
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    move |p: Vec3| Coord { x: p[u], y: p[v] }
}

fn o2(a: Coord<f64>, b: Coord<f64>, c: Coord<f64>) -> i8 {
    let v = orient2d(a, b, c);
    (v > 0.0) as i8 - (v < 0.0) as i8
}

/// Segments fermés [p, q] et [a, b] du plan qui se rencontrent.
fn seg_seg_2d(p: Coord<f64>, q: Coord<f64>, a: Coord<f64>, b: Coord<f64>) -> bool {
    let (d1, d2) = (o2(p, q, a), o2(p, q, b));
    let (d3, d4) = (o2(a, b, p), o2(a, b, q));
    if d1 * d2 < 0 && d3 * d4 < 0 {
        return true;
    }
    let on = |x: Coord<f64>, y: Coord<f64>, z: Coord<f64>| {
        x.x.min(y.x) <= z.x && z.x <= x.x.max(y.x) && x.y.min(y.y) <= z.y && z.y <= x.y.max(y.y)
    };
    (d1 == 0 && on(p, q, a))
        || (d2 == 0 && on(p, q, b))
        || (d3 == 0 && on(a, b, p))
        || (d4 == 0 && on(a, b, q))
}

/// Point `p` du plan dans le triangle fermé (abc), quelle que soit son orientation.
fn point_in_tri_2d(p: Coord<f64>, a: Coord<f64>, b: Coord<f64>, c: Coord<f64>) -> bool {
    let (s1, s2, s3) = (o2(a, b, p), o2(b, c, p), o2(c, a, p));
    let has_neg = s1 < 0 || s2 < 0 || s3 < 0;
    let has_pos = s1 > 0 || s2 > 0 || s3 > 0;
    !(has_neg && has_pos)
}

/// Segment fermé [p, q] et triangle fermé (abc) dans l'espace.
pub fn segment_triangle(p: Vec3, q: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    let (sp, sq) = (o3(a, b, c, p), o3(a, b, c, q));
    if sp * sq > 0 {
        return false;
    }
    if sp == 0 && sq == 0 {
        // Segment dans le plan du triangle.
        let pr = projector(a, b, c);
        let (p2, q2, a2, b2, c2) = (pr(p), pr(q), pr(a), pr(b), pr(c));
        return point_in_tri_2d(p2, a2, b2, c2)
            || point_in_tri_2d(q2, a2, b2, c2)
            || seg_seg_2d(p2, q2, a2, b2)
            || seg_seg_2d(p2, q2, b2, c2)
            || seg_seg_2d(p2, q2, c2, a2);
    }
    // Le segment traverse (ou touche) le plan : la droite (pq) doit passer dans le triangle.
    let (s1, s2, s3) = (o3(p, q, a, b), o3(p, q, b, c), o3(p, q, c, a));
    let has_neg = s1 < 0 || s2 < 0 || s3 < 0;
    let has_pos = s1 > 0 || s2 > 0 || s3 > 0;
    !(has_neg && has_pos)
}

/// Triangles fermés sans sommet commun qui se rencontrent.
pub fn triangles_intersect(t: [Vec3; 3], u: [Vec3; 3]) -> bool {
    (0..3).any(|k| segment_triangle(t[k], t[(k + 1) % 3], u[0], u[1], u[2]))
        || (0..3).any(|k| segment_triangle(u[k], u[(k + 1) % 3], t[0], t[1], t[2]))
}

/// Triangles (v, a, b) et (v, c, d) qui partagent le seul sommet `v` : se rencontrent-ils ailleurs qu'en v ?
pub fn shared_vertex_intersect(v: Vec3, a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> bool {
    let coplanar = o3(v, a, b, c) == 0 && o3(v, a, b, d) == 0;
    if !coplanar {
        // L'intersection est un segment issu de v ; son autre extrémité est sur l'arête opposée à v de l'un
        // des deux triangles (voir le raisonnement dans la documentation du crate).
        return segment_triangle(a, b, v, c, d) || segment_triangle(c, d, v, a, b);
    }
    // Coplanaires : chaque triangle est contenu dans son secteur angulaire en v ; ils se recouvrent si et
    // seulement si les secteurs ouverts se recouvrent, ou si les arêtes opposées se coupent.
    let pr = projector(v, a, b);
    let (v2, a2, b2, c2, d2) = (pr(v), pr(a), pr(b), pr(c), pr(d));
    // Orientation des secteurs : (a, b) et (c, d) parcourus dans le sens direct autour de v.
    let (a2, b2) = if o2(v2, a2, b2) >= 0 {
        (a2, b2)
    } else {
        (b2, a2)
    };
    let (c2, d2) = if o2(v2, c2, d2) >= 0 {
        (c2, d2)
    } else {
        (d2, c2)
    };
    // Direction x strictement dans le secteur (p, q) (secteur < 180°, car triangle non dégénéré).
    let strictly_in =
        |x: Coord<f64>, p: Coord<f64>, q: Coord<f64>| o2(v2, p, x) > 0 && o2(v2, x, q) > 0;
    // Directions confondues (même demi-droite).
    let same_ray = |x: Coord<f64>, y: Coord<f64>| {
        o2(v2, x, y) == 0 && (x.x - v2.x) * (y.x - v2.x) + (x.y - v2.y) * (y.y - v2.y) > 0.0
    };
    strictly_in(c2, a2, b2)
        || strictly_in(d2, a2, b2)
        || strictly_in(a2, c2, d2)
        || strictly_in(b2, c2, d2)
        // Arêtes issues de v superposées le long d'une même demi-droite : contact le long d'un segment.
        || same_ray(a2, c2)
        || same_ray(a2, d2)
        || same_ray(b2, c2)
        || same_ray(b2, d2)
        || seg_seg_2d(a2, b2, c2, d2)
}

/// Triangles (a, b, c) et (b, a, d) qui partagent l'arête (a, b) : se recouvrent-ils (repli à plat) ?
pub fn shared_edge_overlap(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> bool {
    if o3(a, b, c, d) != 0 {
        return false;
    }
    let pr = projector(a, b, c);
    o2(pr(a), pr(b), pr(c)) == o2(pr(a), pr(b), pr(d))
}

/// Triangle d'aire nulle (sommets alignés), au sens exact.
pub fn is_degenerate(a: Vec3, b: Vec3, c: Vec3) -> bool {
    // Alignés si les trois projections sur les plans de coordonnées sont dégénérées.
    let proj = |u: usize, v: usize| {
        let f = |p: Vec3| Coord { x: p[u], y: p[v] };
        o2(f(a), f(b), f(c)) == 0
    };
    proj(0, 1) && proj(1, 2) && proj(2, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn croisement_transverse() {
        let t = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let u = [[0.5, 0.5, -1.0], [0.5, 0.5, 1.0], [3.0, 3.0, 0.0]];
        assert!(triangles_intersect(t, u));
        let w = [[5.0, 5.0, -1.0], [5.0, 5.0, 1.0], [6.0, 6.0, 0.0]];
        assert!(!triangles_intersect(t, w));
    }

    #[test]
    fn coplanaires() {
        let t = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let inside = [[0.2, 0.2, 0.0], [0.5, 0.2, 0.0], [0.2, 0.5, 0.0]];
        let apart = [[3.0, 3.0, 0.0], [4.0, 3.0, 0.0], [3.0, 4.0, 0.0]];
        assert!(triangles_intersect(t, inside));
        assert!(!triangles_intersect(t, apart));
    }

    #[test]
    fn sommet_commun() {
        let v = [0.0, 0.0, 0.0];
        // Éventail plat : deux triangles voisins sans recouvrement.
        assert!(!shared_vertex_intersect(
            v,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0]
        ));
        // Coplanaires qui se recouvrent.
        assert!(shared_vertex_intersect(
            v,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0]
        ));
        // Non coplanaires : le second traverse le premier.
        assert!(shared_vertex_intersect(
            v,
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 0.2, -1.0],
            [0.2, 1.0, 1.0]
        ));
        // Non coplanaires qui ne se touchent qu'en v.
        assert!(!shared_vertex_intersect(
            v,
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0]
        ));
    }

    #[test]
    fn arete_commune_repliee() {
        let (a, b) = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(shared_edge_overlap(a, b, [0.5, 1.0, 0.0], [0.3, 0.5, 0.0]));
        assert!(!shared_edge_overlap(
            a,
            b,
            [0.5, 1.0, 0.0],
            [0.5, -1.0, 0.0]
        ));
        assert!(!shared_edge_overlap(a, b, [0.5, 1.0, 0.0], [0.5, 1.0, 1.0]));
    }

    #[test]
    fn degenere() {
        assert!(is_degenerate([0.0; 3], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]));
        assert!(!is_degenerate([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]));
    }
}
