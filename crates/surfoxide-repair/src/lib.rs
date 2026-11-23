//! `surfoxide-repair` : suppression des auto-intersections en ne déplaçant que des sommets.
//!
//! La connectivité n'est jamais modifiée : la topologie (genre, orientation) est conservée par
//! construction, contrairement aux réparations qui suppriment et rebouchent des faces (MeshFix).
//!
//! - [`safe_update`] : applique de nouvelles positions à une surface saine, en ramenant à leur ancienne
//!   position les sommets des faces qui se coupent, jusqu'à ce qu'il n'y en ait plus. Sert de garde-fou
//!   aux flots (lissage, gonflement vers la piale).
//! - [`repair_intersections`] : lissage laplacien restreint au voisinage des faces fautives, élargi tant
//!   que des intersections subsistent.

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use surfoxide_check::bvh::{Aabb, Bvh};
use surfoxide_check::{
    cross_intersections, cross_intersections_among_in, self_intersections, self_intersections_among_in,
};
use surfoxide_core::adjacency::vertex_neighbors;
use surfoxide_core::vec3::{self, Vec3};
use surfoxide_core::TriMesh;

/// Sommets des faces qui se coupent.
pub fn intersecting_vertices(mesh: &TriMesh) -> HashSet<u32> {
    let faces = mesh.faces();
    self_intersections(mesh)
        .into_iter()
        .flat_map(|(f, g)| faces[f as usize].into_iter().chain(faces[g as usize]))
        .collect()
}

/// Faces à recontrôler après avoir ramené des sommets, et déplacement maximal de ces sommets (mm) : un contrôle
/// de proximité doit regarder les points qui étaient près de ces faces avant le retour, à `margin` près.
///
/// La région porte aussi les hiérarchies de boîtes, construites une fois par appel de [`safe_update_with`] et
/// non à chaque tour : `swept` couvre chaque face dans ses deux positions possibles (tout sommet est soit à
/// son ancienne position, soit à la nouvelle), `obstacle` est la hiérarchie de l'obstacle, construite au
/// premier besoin (un seul obstacle, fixe, par appel).
#[derive(Debug, Clone)]
pub struct Region {
    pub faces: Vec<u32>,
    pub margin: f64,
    swept: Arc<Bvh>,
    obstacle: Arc<OnceLock<Bvh>>,
}

/// Comme [`conflicting_vertices`], restreint aux paires dont une face de `mesh` est dans la région (toutes
/// les faces si `None`).
pub fn conflicting_vertices_among(mesh: &TriMesh, obstacle: Option<&TriMesh>, region: Option<&Region>) -> HashSet<u32> {
    let Some(r) = region else {
        return conflicting_vertices(mesh, obstacle);
    };
    let f = mesh.faces();
    let mut bad: HashSet<u32> = self_intersections_among_in(mesh, &r.swept, &r.faces)
        .into_iter()
        .flat_map(|(a, b)| f[a as usize].into_iter().chain(f[b as usize]))
        .collect();
    if let Some(o) = obstacle {
        let bvh = r.obstacle.get_or_init(|| Bvh::new(o));
        bad.extend(
            cross_intersections_among_in(mesh, o, bvh, &r.faces)
                .into_iter()
                .flat_map(|(a, _)| f[a as usize]),
        );
    }
    bad
}

/// Hiérarchie des faces de `mesh` dont chaque boîte contient la face avec ses sommets aux positions `old`
/// comme aux positions `new`, et dans tout mélange des deux.
fn swept_bvh(mesh: &TriMesh, old: &[Vec3], new: &[Vec3]) -> Bvh {
    let boxes = mesh
        .faces()
        .iter()
        .map(|t| {
            t.iter()
                .fold(Aabb::EMPTY, |b, &i| b.grow(old[i as usize]).grow(new[i as usize]))
        })
        .collect();
    Bvh::from_boxes(boxes)
}

/// Sommets des faces de `mesh` qui se coupent entre elles ou qui coupent `obstacle`.
pub fn conflicting_vertices(mesh: &TriMesh, obstacle: Option<&TriMesh>) -> HashSet<u32> {
    let mut bad = intersecting_vertices(mesh);
    if let Some(o) = obstacle {
        let faces = mesh.faces();
        bad.extend(
            cross_intersections(mesh, o)
                .into_iter()
                .flat_map(|(f, _)| faces[f as usize]),
        );
    }
    bad
}

/// Remplace les positions de `mesh` (supposé sans auto-intersection) par `new`, sauf pour les sommets
/// qu'il faut laisser en place pour ne créer aucune intersection. Rend le nombre de sommets maintenus.
pub fn safe_update(mesh: &mut TriMesh, new: &[Vec3]) -> usize {
    safe_update_against(mesh, new, None)
}

/// Comme [`safe_update`], en interdisant aussi toute intersection avec `obstacle` (que `mesh` ne doit pas
/// couper au départ).
pub fn safe_update_against(mesh: &mut TriMesh, new: &[Vec3], obstacle: Option<&TriMesh>) -> usize {
    safe_update_with(mesh, new, |m, faces| conflicting_vertices_among(m, obstacle, faces))
}

/// Applique les nouvelles positions `new`, puis ramène à leur ancienne position, de proche en proche, les
/// sommets déplacés que `conflicts` signale, jusqu'à ce qu'aucun sommet déplacé ne soit en conflit. Les
/// conflits qui ne touchent que des sommets immobiles existaient déjà : ils sont laissés tels quels, ce qui
/// garantit la terminaison. Rend le nombre de sommets maintenus.
///
/// `conflicts(m, region)` rend les sommets en conflit : sur toute la surface si `region` vaut `None` (premier
/// contrôle), sinon seulement les conflits qui touchent une face de la région. Après le premier contrôle, seuls
/// les sommets ramenés ont bougé : un conflit nouveau ou persistant touche forcément une de leurs faces, et
/// le contrôle restreint suffit (même résultat qu'un contrôle complet, en bien moins de temps).
///
/// Les hiérarchies de boîtes des contrôles restreints sont construites une seule fois (voir [`Region`]) : un
/// tour ne coûte que le voisinage des sommets ramenés, même quand les sommets en conflit se libèrent de proche
/// en proche sur des milliers de tours (pas implicite qui déplace toute la surface).
pub fn safe_update_with(
    mesh: &mut TriMesh,
    new: &[Vec3],
    conflicts: impl Fn(&TriMesh, Option<&Region>) -> HashSet<u32>,
) -> usize {
    assert_eq!(new.len(), mesh.n_vertices());
    let old = mesh.vertices().to_vec();
    mesh.vertices_mut().copy_from_slice(new);
    let vf = surfoxide_core::adjacency::vertex_faces(mesh);
    let mut held: HashSet<u32> = HashSet::new();
    let mut region: Option<Region> = None;
    let mut trees: Option<(Arc<Bvh>, Arc<OnceLock<Bvh>>)> = None;
    loop {
        let fresh: Vec<u32> = conflicts(mesh, region.as_ref())
            .into_iter()
            .filter(|&v| !held.contains(&v) && old[v as usize] != new[v as usize])
            .collect();
        if fresh.is_empty() {
            return held.len();
        }
        let mut faces: Vec<u32> = Vec::new();
        let mut margin: f64 = 0.0;
        for v in fresh {
            margin = margin.max(vec3::dist(old[v as usize], new[v as usize]));
            mesh.vertices_mut()[v as usize] = old[v as usize];
            held.insert(v);
            faces.extend(&vf[v as usize]);
        }
        faces.sort_unstable();
        faces.dedup();
        let (swept, obstacle) = trees
            .get_or_insert_with(|| (Arc::new(swept_bvh(mesh, &old, new)), Arc::new(OnceLock::new())))
            .clone();
        region = Some(Region {
            faces,
            margin,
            swept,
            obstacle,
        });
    }
}

/// Bilan d'une réparation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairReport {
    pub initial: usize,
    pub remaining: usize,
    pub rounds: usize,
}

/// Lisse localement les régions qui se coupent jusqu'à disparition des intersections (au plus
/// `max_rounds` tours). Le voisinage lissé grandit d'un anneau tous les deux tours.
pub fn repair_intersections(mesh: &mut TriMesh, max_rounds: usize) -> RepairReport {
    let nb = vertex_neighbors(mesh);
    let initial = self_intersections(mesh).len();
    let mut remaining = initial;
    let mut rounds = 0;
    while remaining > 0 && rounds < max_rounds {
        let mut region: HashSet<u32> = intersecting_vertices(mesh);
        for _ in 0..1 + rounds / 2 {
            let ring: Vec<u32> = region.iter().flat_map(|&v| nb[v as usize].iter().copied()).collect();
            region.extend(ring);
        }
        for _ in 0..3 {
            let v = mesh.vertices().to_vec();
            for &i in &region {
                let n = &nb[i as usize];
                let mean = n.iter().fold([0.0; 3], |acc, &j| vec3::add(acc, v[j as usize]));
                let mean = vec3::scale(mean, 1.0 / n.len() as f64);
                mesh.vertices_mut()[i as usize] = vec3::lerp(v[i as usize], mean, 0.5);
            }
        }
        rounds += 1;
        remaining = self_intersections(mesh).len();
    }
    RepairReport {
        initial,
        remaining,
        rounds,
    }
}

/// Direction de décalage du sommet `v` : la plus proche de la normale moyenne `n` qui fasse un angle aigu
/// (cosinus ≥ `min_cos`) avec chaque face incidente, trouvée par corrections successives. `None` si la
/// recherche échoue (sommet plissé ou cône trop étroit : la piale bloque alors et la blanche est reprise).
pub fn cone_direction(n: Vec3, face_normals: &[Vec3], min_cos: f64) -> Option<Vec3> {
    let mut d = vec3::normalize(n);
    for _ in 0..100 {
        let (worst, cos) = face_normals
            .iter()
            .map(|&f| (f, vec3::dot(d, f)))
            .fold(([0.0; 3], f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
        if cos >= min_cos {
            return Some(d);
        }
        d = vec3::normalize(vec3::add(d, vec3::scale(worst, min_cos - cos + 0.05)));
    }
    None
}

/// Sommets « anguleux » : la normale moyenne fait un angle de cosinus inférieur à `min_cos` avec l'une des
/// faces incidentes.
pub fn sharp_vertices(mesh: &TriMesh, min_cos: f64) -> Vec<u32> {
    let vf = surfoxide_core::adjacency::vertex_faces(mesh);
    let fnorm = surfoxide_core::adjacency::face_normals(mesh);
    let vnorm = surfoxide_core::adjacency::vertex_normals(mesh);
    (0..mesh.n_vertices() as u32)
        .filter(|&v| {
            vf[v as usize]
                .iter()
                .any(|&f| vec3::dot(vnorm[v as usize], fnorm[f as usize]) < min_cos)
        })
        .collect()
}

/// Lisse (laplacien, pas 1/2) les sommets `seeds` et leurs `rings` premiers anneaux, `iterations` fois.
/// Avec `safe`, aucun pas ne crée d'intersection ; sans, les intersections créées sont réparées ensuite
/// par [`repair_intersections`] (utile pour une lame mince que tout déplacement sûr laisse en place).
pub fn relax_region(mesh: &mut TriMesh, seeds: &[u32], rings: usize, iterations: usize, safe: bool) {
    let nb = vertex_neighbors(mesh);
    let mut region: HashSet<u32> = seeds.iter().copied().collect();
    for _ in 0..rings {
        let ring: Vec<u32> = region.iter().flat_map(|&v| nb[v as usize].iter().copied()).collect();
        region.extend(ring);
    }
    for _ in 0..iterations {
        let v = mesh.vertices().to_vec();
        let mut new = v.clone();
        for &i in &region {
            let n = &nb[i as usize];
            let mean = n.iter().fold([0.0; 3], |acc, &j| vec3::add(acc, v[j as usize]));
            new[i as usize] = vec3::lerp(v[i as usize], vec3::scale(mean, 1.0 / n.len() as f64), 0.5);
        }
        if safe {
            safe_update(mesh, &new);
        } else {
            mesh.vertices_mut().copy_from_slice(&new);
        }
    }
    if !safe {
        repair_intersections(mesh, 30);
    }
}

/// Détend les sommets anguleux (et leur premier anneau) par lissage laplacien local, sans créer
/// d'intersection. Rend le nombre de sommets anguleux restants.
pub fn relax_sharp_vertices(mesh: &mut TriMesh, min_cos: f64, rounds: usize) -> usize {
    let nb = vertex_neighbors(mesh);
    for _ in 0..rounds {
        let sharp = sharp_vertices(mesh, min_cos);
        if sharp.is_empty() {
            return 0;
        }
        let mut region: HashSet<u32> = sharp.iter().copied().collect();
        region.extend(sharp.iter().flat_map(|&v| nb[v as usize].iter().copied()));
        let v = mesh.vertices().to_vec();
        let mut new = v.clone();
        for &i in &region {
            let n = &nb[i as usize];
            let mean = n.iter().fold([0.0; 3], |acc, &j| vec3::add(acc, v[j as usize]));
            new[i as usize] = vec3::lerp(v[i as usize], vec3::scale(mean, 1.0 / n.len() as f64), 0.5);
        }
        safe_update(mesh, &new);
    }
    sharp_vertices(mesh, min_cos).len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;
    use surfoxide_core::topology_report;

    /// Sphère dont un sommet est enfoncé à travers la surface.
    fn dented() -> TriMesh {
        let mut s = icosphere(3, 10.0);
        let p = s.vertices()[0];
        s.vertices_mut()[0] = vec3::scale(p, -1.1);
        s
    }

    #[test]
    fn mise_a_jour_sure() {
        let mut s = icosphere(3, 10.0);
        let mut new = s.vertices().to_vec();
        new[0] = vec3::scale(new[0], -1.3); // créerait des intersections
        new[5] = vec3::scale(new[5], 1.02); // inoffensif
        let held = safe_update(&mut s, &new);
        assert!(held >= 1);
        assert!(self_intersections(&s).is_empty());
        assert_eq!(s.vertices()[5], new[5]);
        assert_ne!(s.vertices()[0], new[0]);
    }

    #[test]
    fn direction_dans_le_cone() {
        let f = [
            vec3::normalize([1.0, 0.0, 1.0]),
            vec3::normalize([-1.0, 0.0, 1.0]),
            vec3::normalize([0.0, 1.0, 0.2]),
        ];
        // Normale moyenne mal orientée vers -y : corrigée.
        let d = cone_direction([0.0, -1.0, 1.0], &f, 0.1).unwrap();
        assert!(f.iter().all(|&n| vec3::dot(d, n) >= 0.1));
        // Faces opposées : pas de direction commune.
        assert!(cone_direction([0.0, 0.0, 1.0], &[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], 0.1).is_none());
    }

    #[test]
    fn reparation_locale() {
        let mut s = dented();
        let before = topology_report(&s);
        assert!(!self_intersections(&s).is_empty());
        let r = repair_intersections(&mut s, 30);
        assert!(r.initial > 0);
        assert_eq!(r.remaining, 0, "{r:?}");
        assert_eq!(topology_report(&s), before);
    }

    /// Générateur congruentiel (graine fixée) : tirages uniformes dans [−1, 1].
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }
    }

    /// Algorithme d'avant la correction, gardé comme oracle : chaque tour reconstruit les hiérarchies de
    /// boîtes de la surface et de l'obstacle (`self_intersections_among`, `cross_intersections_among`).
    fn safe_update_reference(mesh: &mut TriMesh, new: &[Vec3], obstacle: Option<&TriMesh>) -> usize {
        use surfoxide_check::{cross_intersections_among, self_intersections_among};
        let old = mesh.vertices().to_vec();
        mesh.vertices_mut().copy_from_slice(new);
        let vf = surfoxide_core::adjacency::vertex_faces(mesh);
        let mut held: HashSet<u32> = HashSet::new();
        let mut region: Option<Vec<u32>> = None;
        loop {
            let bad: HashSet<u32> = match &region {
                None => conflicting_vertices(mesh, obstacle),
                Some(r) => {
                    let f = mesh.faces();
                    let mut bad: HashSet<u32> = self_intersections_among(mesh, r)
                        .into_iter()
                        .flat_map(|(a, b)| f[a as usize].into_iter().chain(f[b as usize]))
                        .collect();
                    if let Some(o) = obstacle {
                        bad.extend(
                            cross_intersections_among(mesh, o, r)
                                .into_iter()
                                .flat_map(|(a, _)| f[a as usize]),
                        );
                    }
                    bad
                }
            };
            let fresh: Vec<u32> = bad
                .into_iter()
                .filter(|&v| !held.contains(&v) && old[v as usize] != new[v as usize])
                .collect();
            if fresh.is_empty() {
                return held.len();
            }
            let mut faces: Vec<u32> = Vec::new();
            for v in fresh {
                mesh.vertices_mut()[v as usize] = old[v as usize];
                held.insert(v);
                faces.extend(&vf[v as usize]);
            }
            faces.sort_unstable();
            faces.dedup();
            region = Some(faces);
        }
    }

    #[test]
    fn hierarchie_construite_une_fois_meme_resultat_que_l_algorithme_d_origine() {
        // Sphère, puis pas aléatoires de l'ordre de l'arête : beaucoup de conflits, résolus en plusieurs tours ;
        // avec et sans obstacle (sphère intérieure que les pas vers le centre traversent).
        let base = icosphere(4, 10.0);
        let edge = surfoxide_core::adjacency::mean_edge_length(&base);
        let inner = icosphere(3, 9.6);
        let mut rng = Lcg(7);
        let mut tours_multiples = 0;
        for essai in 0..12 {
            let obstacle = (essai % 2 == 1).then_some(&inner);
            let amplitude = if essai < 6 { 0.6 } else { 1.5 } * edge;
            let new: Vec<Vec3> = base
                .vertices()
                .iter()
                .map(|&p| {
                    [
                        p[0] + amplitude * rng.next(),
                        p[1] + amplitude * rng.next(),
                        p[2] + amplitude * rng.next(),
                    ]
                })
                .collect();
            let (mut a, mut b) = (base.clone(), base.clone());
            let held_a = safe_update_against(&mut a, &new, obstacle);
            let held_b = safe_update_reference(&mut b, &new, obstacle);
            assert_eq!(held_a, held_b, "essai {essai}");
            assert_eq!(a.vertices(), b.vertices(), "essai {essai}");
            assert!(self_intersections(&a).is_empty(), "essai {essai}");
            if let Some(o) = obstacle {
                assert!(cross_intersections(&a, o).is_empty(), "essai {essai}");
            }
            tours_multiples += (held_a > 0 && held_a < base.n_vertices()) as usize;
        }
        assert!(tours_multiples >= 8, "{tours_multiples}");
    }

    #[test]
    fn controle_restreint_avec_la_hierarchie_balayee() {
        // La hiérarchie construite sur les boîtes balayées (anciennes et nouvelles positions) donne les mêmes
        // paires que la hiérarchie reconstruite sur la surface du moment, pour des mélanges aléatoires des deux
        // positions, la région étant (comme dans `safe_update_with`) les faces des sommets ramenés.
        let old_mesh = icosphere(3, 10.0);
        let edge = surfoxide_core::adjacency::mean_edge_length(&old_mesh);
        let old = old_mesh.vertices().to_vec();
        let mut rng = Lcg(11);
        let new: Vec<Vec3> = old
            .iter()
            .map(|&p| std::array::from_fn(|a| p[a] + 1.5 * edge * rng.next()))
            .collect();
        let mut m = old_mesh.clone();
        let mut paires = 0;
        // Les deux rôles : surface propre ramenée vers une surface qui se coupe, et l'inverse.
        for (old, new) in [(&old, &new), (&new, &old)] {
            let swept = swept_bvh(&old_mesh, old, new);
            for _ in 0..40 {
                let back: Vec<bool> = (0..old.len()).map(|_| rng.next() < -0.5).collect();
                for (v, &b) in back.iter().enumerate() {
                    m.vertices_mut()[v] = if b { old[v] } else { new[v] };
                }
                let faces: Vec<u32> = (0..m.n_faces() as u32)
                    .filter(|&f| m.faces()[f as usize].iter().any(|&v| back[v as usize]))
                    .collect();
                let attendu = surfoxide_check::self_intersections_among(&m, &faces);
                paires += attendu.len();
                assert_eq!(self_intersections_among_in(&m, &swept, &faces), attendu);
            }
        }
        assert!(paires > 100, "{paires}");
    }
}
