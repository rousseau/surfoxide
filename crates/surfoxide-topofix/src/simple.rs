//! Points simples en connexité (26, 6), par les nombres topologiques de Bertrand et Malandain (1994).
//!
//! Un voxel x est simple pour l'objet X (26-connexe, fond 6-connexe) si et seulement si
//! T26(x, X) = 1 et T̄6(x, X) = 1, où :
//! - T26 est le nombre de composantes 26-connexes de X ∩ N*26(x) ;
//! - T̄6 est le nombre de composantes 6-connexes de X̄ ∩ N*18(x) 6-adjacentes à x.
//!
//! Ajouter ou retirer un point simple ne change ni les composantes, ni les cavités, ni les tunnels.
//! Le voisinage est codé sur 27 bits : bit `(di+1) + 3(dj+1) + 9(dk+1)`, le centre étant le bit 13.

use std::sync::OnceLock;

struct Tables {
    adj26: [u32; 27],
    adj6: [u32; 27],
    n26: u32,
    n18: u32,
    n6: u32,
}

fn offset(b: usize) -> [i32; 3] {
    [(b % 3) as i32 - 1, ((b / 3) % 3) as i32 - 1, (b / 9) as i32 - 1]
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Tables {
            adj26: [0; 27],
            adj6: [0; 27],
            n26: 0,
            n18: 0,
            n6: 0,
        };
        for p in 0..27 {
            let a = offset(p);
            let l1: i32 = a.iter().map(|x| x.abs()).sum();
            if p != 13 {
                t.n26 |= 1 << p;
            }
            if l1 == 1 || l1 == 2 {
                t.n18 |= 1 << p;
            }
            if l1 == 1 {
                t.n6 |= 1 << p;
            }
            for q in 0..27 {
                if p == q {
                    continue;
                }
                let b = offset(q);
                let d: Vec<i32> = (0..3).map(|c| (a[c] - b[c]).abs()).collect();
                if d.iter().all(|&x| x <= 1) {
                    t.adj26[p] |= 1 << q;
                }
                if d.iter().sum::<i32>() == 1 {
                    t.adj6[p] |= 1 << q;
                }
            }
        }
        t
    })
}

/// Nombre de composantes de `set` (adjacence `adj`) qui rencontrent `touch`.
fn components(set: u32, adj: &[u32; 27], touch: u32) -> u32 {
    let mut rest = set;
    let mut n = 0;
    while rest != 0 {
        let mut comp = rest.isolate_lowest_one();
        loop {
            let mut grown = comp;
            let mut bits = comp;
            while bits != 0 {
                let p = bits.trailing_zeros() as usize;
                grown |= adj[p] & set;
                bits &= bits - 1;
            }
            if grown == comp {
                break;
            }
            comp = grown;
        }
        rest &= !comp;
        if comp & touch != 0 {
            n += 1;
        }
    }
    n
}

/// Vrai si le centre du voisinage `nbhd` (27 bits, l'état du centre est ignoré) est un point simple pour
/// l'objet en connexité (26, 6).
pub fn is_simple(nbhd: u32) -> bool {
    let t = tables();
    let obj = nbhd & t.n26;
    if obj == 0 {
        return false; // point isolé
    }
    let bg = !nbhd & t.n18;
    components(obj, &t.adj26, u32::MAX) == 1 && components(bg, &t.adj6, t.n6) == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::volume::IDENTITY;
    use surfoxide_core::Volume;

    /// Générateur pseudo-aléatoire déterministe (xorshift).
    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    /// Définition de référence, par la force brute : le point est simple si l'ajouter à X ∩ N (plongé dans
    /// un volume 5×5×5 de fond) ne change ni le nombre de composantes de l'objet (26), ni celui du fond (6),
    /// ni la caractéristique d'Euler du complexe cubique.
    fn brute(nbhd: u32) -> bool {
        let build = |with_center: bool| {
            let mut v = Volume::filled([5, 5, 5], false, IDENTITY);
            for b in 0..27 {
                let on = if b == 13 { with_center } else { nbhd >> b & 1 == 1 };
                if on {
                    let o = offset(b);
                    v.set((o[0] + 2) as usize, (o[1] + 2) as usize, (o[2] + 2) as usize, true);
                }
            }
            crate::tests::digital_topology(&v)
        };
        build(false) == build(true)
    }

    #[test]
    fn accord_avec_la_force_brute() {
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut simples = 0;
        for n in 0..4000 {
            // Densités variées pour couvrir les cas pleins, creux et intermédiaires.
            let density = n % 7;
            let mut bits = 0u32;
            for b in 0..27 {
                if rng(&mut seed) % 7 < density as u64 {
                    bits |= 1 << b;
                }
            }
            bits &= !(1 << 13);
            assert_eq!(is_simple(bits), brute(bits), "voisinage {bits:027b}");
            simples += is_simple(bits) as usize;
        }
        assert!(simples > 200, "trop peu de points simples testés ({simples})");
    }

    #[test]
    fn cas_elementaires() {
        assert!(!is_simple(0), "point isolé");
        let t = tables();
        assert!(!is_simple(t.n26), "point intérieur : son ajout bouche une cavité");
        // Un seul voisin par une face : simple.
        assert!(is_simple(1 << 12));
        // Deux voisins opposés par une face : l'ajout relierait deux composantes.
        assert!(!is_simple(1 << 12 | 1 << 14));
    }
}
