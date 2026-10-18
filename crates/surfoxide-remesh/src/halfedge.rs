//! Maillage triangulaire fermé modifiable (table des coins) : coupure, fusion et bascule d'arêtes.
//!
//! La demi-arête `h = 3f + k` va du sommet `k` au sommet `k + 1` de la face `f`. Chaque opération
//! réécrit un petit groupe de faces puis recâble les jumelles : celles internes au groupe sont retrouvées
//! entre elles, celles vers l'extérieur sont reprises de l'état d'avant l'opération.

use surfoxide_core::vec3::{self, Vec3};
use surfoxide_core::TriMesh;

const NONE: u32 = u32::MAX;

#[derive(Debug, Clone)]
pub struct HalfEdgeMesh {
    pub pos: Vec<Vec3>,
    pub faces: Vec<[u32; 3]>,
    pub face_alive: Vec<bool>,
    pub vertex_alive: Vec<bool>,
    /// Jumelle de chaque demi-arête.
    twin: Vec<u32>,
    /// Une demi-arête sortante de chaque sommet.
    out: Vec<u32>,
}

/// Le maillage n'est pas une surface fermée 2-variété.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotClosedManifold;

impl HalfEdgeMesh {
    pub fn from_mesh(mesh: &TriMesh) -> Result<Self, NotClosedManifold> {
        let faces = mesh.faces().to_vec();
        let n = faces.len() * 3;
        let mut twin = vec![NONE; n];
        let mut map: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::with_capacity(n);
        for (f, t) in faces.iter().enumerate() {
            for k in 0..3 {
                let h = (3 * f + k) as u32;
                if map.insert((t[k], t[(k + 1) % 3]), h).is_some() {
                    return Err(NotClosedManifold);
                }
            }
        }
        for (&(a, b), &h) in &map {
            match map.get(&(b, a)) {
                Some(&g) => twin[h as usize] = g,
                None => return Err(NotClosedManifold),
            }
        }
        let mut out = vec![NONE; mesh.n_vertices()];
        for (f, t) in faces.iter().enumerate() {
            for k in 0..3 {
                out[t[k] as usize] = (3 * f + k) as u32;
            }
        }
        if out.contains(&NONE) {
            return Err(NotClosedManifold);
        }
        Ok(Self {
            pos: mesh.vertices().to_vec(),
            face_alive: vec![true; faces.len()],
            vertex_alive: vec![true; mesh.n_vertices()],
            faces,
            twin,
            out,
        })
    }

    /// Maillage compacté (sommets et faces vivants renumérotés).
    pub fn to_mesh(&self) -> TriMesh {
        let mut map = vec![NONE; self.pos.len()];
        let mut pos = Vec::new();
        for (v, &alive) in self.vertex_alive.iter().enumerate() {
            if alive {
                map[v] = pos.len() as u32;
                pos.push(self.pos[v]);
            }
        }
        let faces = self
            .faces
            .iter()
            .zip(&self.face_alive)
            .filter(|(_, &a)| a)
            .map(|(f, _)| f.map(|v| map[v as usize]))
            .collect();
        TriMesh::new(pos, faces).expect("maillage cohérent")
    }

    #[inline]
    pub fn from(&self, h: u32) -> u32 {
        self.faces[(h / 3) as usize][(h % 3) as usize]
    }

    #[inline]
    pub fn to(&self, h: u32) -> u32 {
        self.faces[(h / 3) as usize][((h % 3 + 1) % 3) as usize]
    }

    #[inline]
    pub fn next(&self, h: u32) -> u32 {
        3 * (h / 3) + (h % 3 + 1) % 3
    }

    #[inline]
    pub fn prev(&self, h: u32) -> u32 {
        3 * (h / 3) + (h % 3 + 2) % 3
    }

    #[inline]
    pub fn twin(&self, h: u32) -> u32 {
        self.twin[h as usize]
    }

    /// Demi-arêtes sortantes de `v`.
    pub fn outgoing(&self, v: u32) -> Vec<u32> {
        let start = self.out[v as usize];
        let mut out = vec![start];
        // prev(h) arrive en v ; sa jumelle part de v : on tourne autour de v.
        let mut h = self.twin(self.prev(start));
        while h != start {
            out.push(h);
            h = self.twin(self.prev(h));
            if out.len() > 1000 {
                panic!("voisinage de sommet incohérent");
            }
        }
        out
    }

    pub fn neighbors(&self, v: u32) -> Vec<u32> {
        self.outgoing(v).into_iter().map(|h| self.to(h)).collect()
    }

    pub fn valence(&self, v: u32) -> usize {
        self.outgoing(v).len()
    }

    pub fn face_normal(&self, f: usize) -> Vec3 {
        let t = self.faces[f];
        vec3::triangle_normal(
            self.pos[t[0] as usize],
            self.pos[t[1] as usize],
            self.pos[t[2] as usize],
        )
    }

    /// Normale (non normalisée, pondérée par l'aire) au sommet `v`.
    pub fn vertex_normal(&self, v: u32) -> Vec3 {
        self.outgoing(v).into_iter().fold([0.0; 3], |acc, h| {
            vec3::add(acc, self.face_normal((h / 3) as usize))
        })
    }

    pub fn edge_length(&self, h: u32) -> f64 {
        vec3::dist(
            self.pos[self.from(h) as usize],
            self.pos[self.to(h) as usize],
        )
    }

    /// Réécrit les faces `group` (déjà modifiées, sauf celles listées dans `killed`) et recâble les
    /// jumelles. `external` : jumelles extérieures relevées avant la modification, par (départ, arrivée)
    /// de la demi-arête intérieure.
    fn relink(&mut self, group: &[usize], external: &[((u32, u32), u32)]) {
        let mut local: Vec<((u32, u32), u32)> = Vec::with_capacity(group.len() * 3);
        for &f in group {
            if !self.face_alive[f] {
                continue;
            }
            for k in 0..3 {
                let h = (3 * f + k) as u32;
                local.push(((self.from(h), self.to(h)), h));
            }
        }
        for &((a, b), h) in &local {
            let t = local
                .iter()
                .find(|(k, _)| *k == (b, a))
                .map(|&(_, g)| g)
                .or_else(|| external.iter().find(|(k, _)| *k == (a, b)).map(|&(_, g)| g))
                .expect("jumelle introuvable");
            self.twin[h as usize] = t;
            self.twin[t as usize] = h;
            self.out[a as usize] = h;
        }
    }

    /// Jumelles extérieures au groupe de faces `group`.
    fn externals(&self, group: &[usize]) -> Vec<((u32, u32), u32)> {
        let mut ext = Vec::new();
        for &f in group {
            for k in 0..3 {
                let h = (3 * f + k) as u32;
                let t = self.twin(h);
                if !group.contains(&((t / 3) as usize)) {
                    ext.push(((self.from(h), self.to(h)), t));
                }
            }
        }
        ext
    }

    fn add_face(&mut self, t: [u32; 3]) -> usize {
        self.faces.push(t);
        self.face_alive.push(true);
        self.twin.extend([NONE; 3]);
        self.faces.len() - 1
    }

    /// Coupe l'arête `h` en son milieu ; rend le nouveau sommet.
    pub fn split(&mut self, h: u32) -> u32 {
        let g = self.twin(h);
        let (f1, f2) = ((h / 3) as usize, (g / 3) as usize);
        let (a, b) = (self.from(h), self.to(h));
        let c = self.to(self.next(h));
        let d = self.to(self.next(g));
        let ext = self.externals(&[f1, f2]);
        let m = self.pos.len() as u32;
        self.pos
            .push(vec3::lerp(self.pos[a as usize], self.pos[b as usize], 0.5));
        self.vertex_alive.push(true);
        self.out.push(NONE);
        // (a, b, c) → (a, m, c) + (m, b, c) ; (b, a, d) → (m, a, d) + (b, m, d).
        self.faces[f1] = [a, m, c];
        self.faces[f2] = [m, a, d];
        let f3 = self.add_face([m, b, c]);
        let f4 = self.add_face([b, m, d]);
        self.relink(&[f1, f2, f3, f4], &ext);
        m
    }

    /// Bascule l'arête `h` (a→b) entre (a, b, c) et (b, a, d) en (c, d) ; refusée si c et d sont déjà
    /// voisins.
    pub fn flip(&mut self, h: u32) -> bool {
        let g = self.twin(h);
        let (f1, f2) = ((h / 3) as usize, (g / 3) as usize);
        let (a, b) = (self.from(h), self.to(h));
        let c = self.to(self.next(h));
        let d = self.to(self.next(g));
        if c == d || self.neighbors(c).contains(&d) {
            return false;
        }
        let ext = self.externals(&[f1, f2]);
        self.faces[f1] = [a, d, c];
        self.faces[f2] = [d, b, c];
        self.relink(&[f1, f2], &ext);
        true
    }

    /// Vrai si la fusion de l'arête `h` (a→b) préserve la topologie : les seuls voisins communs de a et b
    /// sont les deux sommets opposés, qui gardent au moins trois voisins.
    pub fn can_collapse(&self, h: u32) -> bool {
        let g = self.twin(h);
        let (a, b) = (self.from(h), self.to(h));
        let c = self.to(self.next(h));
        let d = self.to(self.next(g));
        let na = self.neighbors(a);
        let nb = self.neighbors(b);
        let common = na.iter().filter(|x| nb.contains(x)).count();
        common == 2 && self.valence(c) > 3 && self.valence(d) > 3 && na.len() + nb.len() > 6
    }

    /// Fusionne l'arête `h` (a→b) : a disparaît, b est placé en `p`. À n'appeler qu'après
    /// [`Self::can_collapse`].
    pub fn collapse(&mut self, h: u32, p: Vec3) {
        let g = self.twin(h);
        let (f1, f2) = ((h / 3) as usize, (g / 3) as usize);
        let (a, b) = (self.from(h), self.to(h));
        let group: Vec<usize> = self
            .outgoing(a)
            .into_iter()
            .map(|e| (e / 3) as usize)
            .collect();
        let ext = self.externals(&group);
        self.face_alive[f1] = false;
        self.face_alive[f2] = false;
        for &f in &group {
            for v in &mut self.faces[f] {
                if *v == a {
                    *v = b;
                }
            }
        }
        self.vertex_alive[a as usize] = false;
        self.pos[b as usize] = p;
        // Les arêtes extérieures de f1 (b–c) et de f2 (d–b) sont reprises, avec la même clé, par les faces
        // voisines où a vient d'être renommé en b : le recâblage les retrouve par (départ, arrivée).
        let live: Vec<usize> = group
            .iter()
            .copied()
            .filter(|&f| self.face_alive[f])
            .collect();
        self.relink(&live, &ext);
        for &f in &live {
            for &v in &self.faces[f] {
                if self.vertex_alive[v as usize] {
                    let k = self.faces[f].iter().position(|&w| w == v).unwrap();
                    self.out[v as usize] = (3 * f + k) as u32;
                }
            }
        }
    }

    /// Demi-arêtes vivantes, une par arête (celle de plus petit indice).
    pub fn edges(&self) -> Vec<u32> {
        (0..self.faces.len() as u32 * 3)
            .filter(|&h| self.face_alive[(h / 3) as usize] && h < self.twin(h))
            .collect()
    }

    /// Sommets vivants.
    pub fn vertices(&self) -> Vec<u32> {
        (0..self.pos.len() as u32)
            .filter(|&v| self.vertex_alive[v as usize])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;
    use surfoxide_core::topology_report;

    fn check(m: &HalfEdgeMesh) {
        let t = m.to_mesh();
        let r = topology_report(&t);
        assert!(r.is_sphere_like(), "{r:?}");
        // Les jumelles sont cohérentes.
        for h in m.edges() {
            let g = m.twin(h);
            assert_eq!(m.twin(g), h);
            assert_eq!((m.from(g), m.to(g)), (m.to(h), m.from(h)));
        }
    }

    #[test]
    fn coupures_bascules_fusions_preservent_la_sphere() {
        let mut m = HalfEdgeMesh::from_mesh(&icosphere(2, 1.0)).unwrap();
        for h in m.edges().into_iter().step_by(3) {
            m.split(h);
        }
        check(&m);
        let mut flips = 0;
        for h in m.edges().into_iter().step_by(5) {
            flips += m.flip(h) as usize;
        }
        assert!(flips > 0);
        check(&m);
        let mut collapses = 0;
        for h in m.edges().into_iter().step_by(7) {
            if m.face_alive[(h / 3) as usize] && m.can_collapse(h) {
                let p = m.pos[m.to(h) as usize];
                m.collapse(h, p);
                collapses += 1;
                check(&m);
            }
        }
        assert!(collapses > 0);
    }
}
