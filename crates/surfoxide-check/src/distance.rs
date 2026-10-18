//! Distances point–triangle et indice d'enroulement.

use surfoxide_core::vec3::{self, Vec3};

/// Point du triangle (abc) le plus proche de `p` (Ericson, *Real-Time Collision Detection*, § 5.1.5).
pub fn closest_point_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let (ab, ac, ap) = (vec3::sub(b, a), vec3::sub(c, a), vec3::sub(p, a));
    let (d1, d2) = (vec3::dot(ab, ap), vec3::dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = vec3::sub(p, b);
    let (d3, d4) = (vec3::dot(ab, bp), vec3::dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return vec3::add(a, vec3::scale(ab, d1 / (d1 - d3)));
    }
    let cp = vec3::sub(p, c);
    let (d5, d6) = (vec3::dot(ab, cp), vec3::dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return vec3::add(a, vec3::scale(ac, d2 / (d2 - d6)));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return vec3::add(b, vec3::scale(vec3::sub(c, b), w));
    }
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    vec3::add(a, vec3::add(vec3::scale(ab, v), vec3::scale(ac, w)))
}

/// Distance au carré de `p` au triangle (abc).
pub fn distance2_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let q = closest_point_triangle(p, a, b, c);
    let d = vec3::sub(p, q);
    vec3::dot(d, d)
}

/// Angle solide (sr) sous lequel le triangle orienté (abc) est vu depuis `p` (Van Oosterom & Strackee).
pub fn solid_angle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let (a, b, c) = (vec3::sub(a, p), vec3::sub(b, p), vec3::sub(c, p));
    let (la, lb, lc) = (vec3::norm(a), vec3::norm(b), vec3::norm(c));
    let num = vec3::dot(a, vec3::cross(b, c));
    let den = la * lb * lc + vec3::dot(a, b) * lc + vec3::dot(a, c) * lb + vec3::dot(b, c) * la;
    2.0 * num.atan2(den)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plus_proche_point() {
        let (a, b, c) = ([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]);
        assert_eq!(
            closest_point_triangle([0.5, 0.5, 3.0], a, b, c),
            [0.5, 0.5, 0.0]
        );
        assert_eq!(closest_point_triangle([-1.0, -1.0, 0.0], a, b, c), a);
        assert_eq!(
            closest_point_triangle([1.0, -1.0, 0.0], a, b, c),
            [1.0, 0.0, 0.0]
        );
        assert!((distance2_triangle([2.0, 2.0, 0.0], a, b, c) - 2.0).abs() < 1e-12);
    }
}
