//! Matrice creuse (CSR) et gradient conjugué préconditionné par la diagonale.

/// Matrice creuse carrée au format CSR.
#[derive(Debug, Clone)]
pub struct Csr {
    n: usize,
    row_start: Vec<usize>,
    cols: Vec<u32>,
    vals: Vec<f64>,
}

impl Csr {
    /// Assemble une matrice `n × n` à partir de triplets (i, j, v) ; les doublons sont additionnés.
    pub fn from_triplets(n: usize, mut t: Vec<(u32, u32, f64)>) -> Self {
        t.sort_unstable_by_key(|&(i, j, _)| (i, j));
        let mut row_start = vec![0usize; n + 1];
        let mut cols: Vec<u32> = Vec::with_capacity(t.len());
        let mut vals: Vec<f64> = Vec::with_capacity(t.len());
        let mut last: Option<(u32, u32)> = None;
        for (i, j, v) in t {
            if last == Some((i, j)) {
                *vals.last_mut().unwrap() += v;
            } else {
                cols.push(j);
                vals.push(v);
                row_start[i as usize + 1] += 1;
                last = Some((i, j));
            }
        }
        for i in 0..n {
            row_start[i + 1] += row_start[i];
        }
        Self {
            n,
            row_start,
            cols,
            vals,
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Diagonale de la matrice.
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.n)
            .map(|i| {
                let r = self.row_start[i]..self.row_start[i + 1];
                self.cols[r.clone()]
                    .iter()
                    .zip(&self.vals[r])
                    .find(|(&j, _)| j as usize == i)
                    .map_or(0.0, |(_, &v)| v)
            })
            .collect()
    }

    /// Produit matrice–vecteur `y = A x`.
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|i| {
                let r = self.row_start[i]..self.row_start[i + 1];
                self.cols[r.clone()]
                    .iter()
                    .zip(&self.vals[r])
                    .map(|(&j, &v)| v * x[j as usize])
                    .sum()
            })
            .collect()
    }

    /// Combinaison linéaire `a · self + b · other` (même structure non requise).
    pub fn add_scaled(&self, a: f64, other: &Csr, b: f64) -> Csr {
        assert_eq!(self.n, other.n, "dimensions incompatibles");
        let mut t = Vec::with_capacity(self.vals.len() + other.vals.len());
        for (m, s) in [(self, a), (other, b)] {
            for i in 0..m.n {
                for k in m.row_start[i]..m.row_start[i + 1] {
                    t.push((i as u32, m.cols[k], s * m.vals[k]));
                }
            }
        }
        Csr::from_triplets(self.n, t)
    }
}

/// Résultat d'une résolution itérative.
#[derive(Debug, Clone)]
pub struct Solve {
    pub x: Vec<f64>,
    pub iterations: usize,
    /// Résidu relatif final ‖b − Ax‖ / ‖b‖.
    pub residual: f64,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Gradient conjugué préconditionné par la diagonale (Jacobi), pour `A` symétrique définie positive.
pub fn conjugate_gradient(a: &Csr, b: &[f64], tol: f64, max_iter: usize) -> Solve {
    let n = a.n();
    let inv_d: Vec<f64> = a
        .diagonal()
        .iter()
        .map(|&d| if d != 0.0 { 1.0 / d } else { 1.0 })
        .collect();
    let norm_b = dot(b, b).sqrt();
    let mut x = vec![0.0; n];
    if norm_b == 0.0 {
        return Solve {
            x,
            iterations: 0,
            residual: 0.0,
        };
    }
    let mut r = b.to_vec();
    let mut z: Vec<f64> = r.iter().zip(&inv_d).map(|(r, d)| r * d).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut residual = 1.0;
    let mut it = 0;
    while it < max_iter {
        let ap = a.mul_vec(&p);
        let step = rz / dot(&p, &ap);
        for i in 0..n {
            x[i] += step * p[i];
            r[i] -= step * ap[i];
        }
        it += 1;
        residual = dot(&r, &r).sqrt() / norm_b;
        if residual < tol {
            break;
        }
        for i in 0..n {
            z[i] = r[i] * inv_d[i];
        }
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    Solve {
        x,
        iterations: it,
        residual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assemblage_et_produit() {
        let a = Csr::from_triplets(2, vec![(0, 0, 1.0), (0, 1, 2.0), (0, 0, 3.0), (1, 1, 5.0)]);
        assert_eq!(a.mul_vec(&[1.0, 1.0]), vec![6.0, 5.0]);
        assert_eq!(a.diagonal(), vec![4.0, 5.0]);
    }

    #[test]
    fn gradient_conjugue_sur_laplacien_1d() {
        // Matrice tridiagonale (2, −1) + 0,1·I : symétrique définie positive.
        let n = 50;
        let mut t = Vec::new();
        for i in 0..n as u32 {
            t.push((i, i, 2.1));
            if i > 0 {
                t.push((i, i - 1, -1.0));
                t.push((i - 1, i, -1.0));
            }
        }
        let a = Csr::from_triplets(n, t);
        let x_true: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let b = a.mul_vec(&x_true);
        let s = conjugate_gradient(&a, &b, 1e-12, 1000);
        assert!(s.residual < 1e-12);
        for (x, y) in s.x.iter().zip(&x_true) {
            assert!((x - y).abs() < 1e-9);
        }
    }
}
