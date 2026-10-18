//! Lissage de Taubin (1995) : alternance d'un pas laplacien λ > 0 (lissage) et d'un pas μ < −λ
//! (regonflement), filtre passe-bas sans le rétrécissement du lissage laplacien seul. Seules les positions
//! changent : la topologie est conservée.

use surfoxide_core::adjacency::vertex_neighbors;
use surfoxide_core::vec3;
use surfoxide_core::TriMesh;

/// `iterations` paires de pas (λ, μ) avec le laplacien uniforme (moyenne des voisins).
pub fn taubin(mesh: &mut TriMesh, iterations: usize, lambda: f64, mu: f64) {
    let nb = vertex_neighbors(mesh);
    let mut next = mesh.vertices().to_vec();
    for _ in 0..iterations {
        for factor in [lambda, mu] {
            let v = mesh.vertices();
            for (i, n) in nb.iter().enumerate() {
                if n.is_empty() {
                    continue;
                }
                let mean = n
                    .iter()
                    .fold([0.0; 3], |acc, &j| vec3::add(acc, v[j as usize]));
                let mean = vec3::scale(mean, 1.0 / n.len() as f64);
                next[i] = vec3::add(v[i], vec3::scale(vec3::sub(mean, v[i]), factor));
            }
            mesh.vertices_mut().copy_from_slice(&next);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;

    #[test]
    fn sphere_bruitee_lissee_sans_retrecir() {
        let mut s = icosphere(4, 10.0);
        let v0 = s.signed_volume();
        // Bruit radial déterministe.
        for (i, p) in s.vertices_mut().iter_mut().enumerate() {
            let f = 1.0 + 0.03 * ((i * 7919 % 101) as f64 / 50.0 - 1.0);
            *p = vec3::scale(*p, f);
        }
        let rough = |m: &TriMesh| {
            m.vertices()
                .iter()
                .map(|p| (vec3::norm(*p) - 10.0).abs())
                .sum::<f64>()
                / m.n_vertices() as f64
        };
        let before = rough(&s);
        taubin(&mut s, 10, 0.5, -0.53);
        assert!(rough(&s) < before / 3.0, "{} contre {before}", rough(&s));
        assert!((s.signed_volume() - v0).abs() / v0 < 0.02);
    }
}
