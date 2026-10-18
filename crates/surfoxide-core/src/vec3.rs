//! Opérations élémentaires sur les vecteurs 3D représentés par `[f64; 3]`.

pub type Vec3 = [f64; 3];

#[inline]
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
pub fn dist(a: Vec3, b: Vec3) -> f64 {
    norm(sub(a, b))
}

/// Vecteur unitaire de même direction ; le vecteur nul est rendu tel quel.
#[inline]
pub fn normalize(a: Vec3) -> Vec3 {
    let n = norm(a);
    if n > 0.0 {
        scale(a, 1.0 / n)
    } else {
        a
    }
}

/// Combinaison `a + t (b − a)`.
#[inline]
pub fn lerp(a: Vec3, b: Vec3, t: f64) -> Vec3 {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

/// Normale non normalisée du triangle (a, b, c), de norme égale au double de son aire.
#[inline]
pub fn triangle_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    cross(sub(b, a), sub(c, a))
}
