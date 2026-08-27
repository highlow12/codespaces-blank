//! Browser-safe numerical kernels. Matrices are flat, row-major f32 buffers.
//! Calls are synchronous; a worker terminates between tiled calls to cancel.

use serde::Serialize;
use wasm_bindgen::prelude::*;

const EPS: f32 = 1e-12;
fn err(s: impl Into<String>) -> JsValue { JsValue::from_str(&s.into()) }
fn product(a: usize, b: usize, label: &str) -> Result<usize, JsValue> { a.checked_mul(b).ok_or_else(|| err(format!("{label} size overflows usize"))) }
fn matrix(v: &[f32], rows: usize, cols: usize, label: &str) -> Result<(), JsValue> {
    if rows == 0 || cols == 0 { return Err(err(format!("{label} needs positive rows and columns"))); }
    if v.len() != product(rows, cols, label)? { return Err(err(format!("{label} length does not match {rows}x{cols}"))); }
    if v.iter().any(|x| !x.is_finite()) { return Err(err(format!("{label} has non-finite values"))); }
    Ok(())
}
fn square(v: &[f32], n: usize, label: &str) -> Result<(), JsValue> {
    matrix(v, n, n, label)?;
    if v.iter().any(|x| *x < 0.0) { return Err(err(format!("{label} has a negative distance"))); }
    Ok(())
}
fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }
fn cosine(a: &[f32], b: &[f32]) -> f32 { let n = dot(a,a).sqrt()*dot(b,b).sqrt(); if n <= EPS { 1.0 } else { (1.0-dot(a,b)/n).clamp(0.0,2.0) } }
fn euclidean(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x,y)| { let delta=x-y; delta*delta }).sum::<f32>().sqrt() }
fn normalize_slice(a: &mut [f32]) { let n = dot(a,a).sqrt(); if n > EPS { for x in a { *x /= n; } } }

#[wasm_bindgen]
pub fn normalize(rows: Vec<f32>, row_count: usize, dimension: usize) -> Result<Vec<f32>, JsValue> {
    matrix(&rows,row_count,dimension,"rows")?; let mut out=rows;
    for row in out.chunks_exact_mut(dimension) { normalize_slice(row); } Ok(out)
}

#[wasm_bindgen]
pub fn matmul(a: Vec<f32>, b: Vec<f32>, m: usize, k: usize, n: usize) -> Result<Vec<f32>, JsValue> {
    matrix(&a,m,k,"left matrix")?; matrix(&b,k,n,"right matrix")?;
    let mut out=vec![0.0;product(m,n,"matrix product")?];
    for i in 0..m { for p in 0..k { let v=a[i*k+p]; for j in 0..n { out[i*n+j]+=v*b[p*n+j]; } } } Ok(out)
}

#[derive(Serialize)]
struct PcaResult { projected: Vec<f32>, basis: Vec<f32>, mean: Vec<f32>, explained: Vec<f32>, components: usize }
fn xorshift(s: &mut u32) -> u32 { *s^=*s<<13; *s^=*s>>17; *s^=*s<<5; *s }
fn random(s: &mut u32) -> f32 { xorshift(s) as f32/u32::MAX as f32*2.0-1.0 }
fn orthogonal(q: &mut [f32], rows:usize, cols:usize) {
    for c in 0..cols { for p in 0..c { let mut d=0.0; for r in 0..rows {d+=q[r*cols+c]*q[r*cols+p];} for r in 0..rows {q[r*cols+c]-=d*q[r*cols+p];} }
        let n=(0..rows).map(|r|q[r*cols+c]*q[r*cols+c]).sum::<f32>().sqrt();
        if n>EPS { for r in 0..rows {q[r*cols+c]/=n;} } else if c<rows { for r in 0..rows{q[r*cols+c]=0.0;} q[c*cols+c]=1.0; }
    }
}

/// Deterministic randomized PCA/SVD. No d² covariance matrix is allocated.
#[wasm_bindgen]
pub fn randomized_pca(rows: Vec<f32>, row_count:usize, dimension:usize, wanted:usize, oversamples:usize, power_iterations:usize, seed:u32) -> Result<JsValue,JsValue> {
    matrix(&rows,row_count,dimension,"rows")?; if wanted==0{return Err(err("PCA components must be positive"));}
    let components=wanted.min(row_count).min(dimension); let sketch=components.saturating_add(oversamples).min(row_count).min(dimension);
    let mut mean=vec![0.0;dimension]; for r in rows.chunks_exact(dimension){for (i,x) in r.iter().enumerate(){mean[i]+=*x/row_count as f32;}}
    let mut x=rows; for r in x.chunks_exact_mut(dimension){for(i,v)in r.iter_mut().enumerate(){*v-=mean[i];}}
    let mut state=seed.max(1); let mut omega=vec![0.0;product(dimension,sketch,"PCA projection")?];for z in &mut omega{*z=random(&mut state);}
    let mut q=vec![0.0;product(row_count,sketch,"PCA sketch")?];
    for r in 0..row_count{for c in 0..sketch{for d in 0..dimension{q[r*sketch+c]+=x[r*dimension+d]*omega[d*sketch+c];}}}orthogonal(&mut q,row_count,sketch);
    for _ in 0..power_iterations.min(8){let mut z=vec![0.0;product(dimension,sketch,"PCA iteration")?];for d in 0..dimension{for c in 0..sketch{for r in 0..row_count{z[d*sketch+c]+=x[r*dimension+d]*q[r*sketch+c];}}}for r in 0..row_count{for c in 0..sketch{q[r*sketch+c]=0.;for d in 0..dimension{q[r*sketch+c]+=x[r*dimension+d]*z[d*sketch+c];}}}orthogonal(&mut q,row_count,sketch);}
    let mut b=vec![0.;product(sketch,dimension,"PCA reduced matrix")?];for c in 0..sketch{for d in 0..dimension{for r in 0..row_count{b[c*dimension+d]+=q[r*sketch+c]*x[r*dimension+d];}}}
    let mut gram=vec![0.;product(sketch,sketch,"PCA gram matrix")?];for i in 0..sketch{for j in i..sketch{let z=dot(&b[i*dimension..(i+1)*dimension],&b[j*dimension..(j+1)*dimension]);gram[i*sketch+j]=z;gram[j*sketch+i]=z;}}
    let(mut basis,mut explained)=(Vec::with_capacity(components*dimension),Vec::with_capacity(components));
    for component in 0..components{let mut v=vec![0.;sketch];v[component%sketch]=1.;for _ in 0..48{let mut next=vec![0.;sketch];for i in 0..sketch{for j in 0..sketch{next[i]+=gram[i*sketch+j]*v[j];}}let n=dot(&next,&next).sqrt();if n<=EPS{break;}for z in &mut next{*z/=n;}v=next;}let mut gv=vec![0.;sketch];for i in 0..sketch{for j in 0..sketch{gv[i]+=gram[i*sketch+j]*v[j];}}let eigen=dot(&v,&gv).max(0.);let root=eigen.sqrt();let mut axis=vec![0.;dimension];if root>EPS{for d in 0..dimension{for c in 0..sketch{axis[d]+=b[c*dimension+d]*v[c]/root;}}normalize_slice(&mut axis);}for i in 0..sketch{for j in 0..sketch{gram[i*sketch+j]-=eigen*v[i]*v[j];}}basis.extend(axis);explained.push(eigen/row_count.saturating_sub(1).max(1)as f32);}
    let mut projected=vec![0.;product(row_count,components,"PCA output")?];for r in 0..row_count{for c in 0..components{projected[r*components+c]=dot(&x[r*dimension..(r+1)*dimension],&basis[c*dimension..(c+1)*dimension]);}}
    serde_wasm_bindgen::to_value(&PcaResult{projected,basis,mean,explained,components}).map_err(|e|err(e.to_string()))
}
#[wasm_bindgen] pub fn pca(rows:Vec<f32>,n:usize,d:usize,c:usize)->Result<JsValue,JsValue>{randomized_pca(rows,n,d,c,8,2,42)}

#[wasm_bindgen]
pub fn cosine_distances_tiled(rows:Vec<f32>,n:usize,d:usize,tile:usize)->Result<Vec<f32>,JsValue>{matrix(&rows,n,d,"rows")?;let mut out=vec![0.;product(n,n,"cosine matrix")?];let t=tile.max(1).min(n);for ii in(0..n).step_by(t){for jj in(0..n).step_by(t){for i in ii..(ii+t).min(n){for j in jj..(jj+t).min(n){out[i*n+j]=if i==j{0.}else{cosine(&rows[i*d..(i+1)*d],&rows[j*d..(j+1)*d])};}}}}Ok(out)}
#[wasm_bindgen] pub fn cosine_distances(rows:Vec<f32>,n:usize,d:usize,t:usize)->Result<Vec<f32>,JsValue>{cosine_distances_tiled(rows,n,d,t)}

#[derive(Serialize)] struct KnnResult{indices:Vec<u32>,distances:Vec<f32>,rows:usize,k:usize}
fn rank_distances(distances:&[f32],n:usize,k:usize)->Result<Vec<u32>,JsValue>{square(distances,n,"distance matrix")?;if k==0||k>=n{return Err(err("k must be between 1 and row_count - 1"));}let mut out=Vec::with_capacity(product(n,k,"kNN")?);for i in 0..n{let mut c:(Vec<(f32,usize)>)=(0..n).filter(|j|*j!=i).map(|j|(distances[i*n+j],j)).collect();c.sort_unstable_by(|a,b|a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));out.extend(c.into_iter().take(k).map(|(_,j)|j as u32));}Ok(out)}
#[wasm_bindgen] pub fn exact_knn(distances:Vec<f32>,n:usize,k:usize)->Result<Vec<u32>,JsValue>{rank_distances(&distances,n,k)}

/// Tiled all-pairs comparison that retains O(nk), not O(n²), output.
#[wasm_bindgen]
pub fn exact_knn_cosine_tiled(rows:Vec<f32>,n:usize,d:usize,k:usize,tile:usize)->Result<JsValue,JsValue>{matrix(&rows,n,d,"rows")?;if k==0||k>=n{return Err(err("k must be between 1 and row_count - 1"));}let t=tile.max(1).min(n);let mut indices=vec![0;product(n,k,"kNN indices")?];let mut distances=vec![f32::INFINITY;product(n,k,"kNN distances")?];for start in(0..n).step_by(t){let end=(start+t).min(n);let mut best:Vec<Vec<(f32,usize)>>=(start..end).map(|_|Vec::with_capacity(k+t)).collect();for target in(0..n).step_by(t){for i in start..end{for j in target..(target+t).min(n){if i!=j{best[i-start].push((cosine(&rows[i*d..(i+1)*d],&rows[j*d..(j+1)*d]),j));}}best[i-start].sort_unstable_by(|a,b|a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));best[i-start].truncate(k);}}for i in start..end{for(rank,(distance,index))in best[i-start].iter().enumerate(){indices[i*k+rank]=*index as u32;distances[i*k+rank]=*distance;}}}serde_wasm_bindgen::to_value(&KnnResult{indices,distances,rows:n,k}).map_err(|e|err(e.to_string()))}

fn level(index:usize,seed:u32)->usize{let mut s=seed.max(1)^(index as u32).wrapping_mul(0x9e3779b9);let mut out=0;while xorshift(&mut s)&1==0&&out<16{out+=1;}out}
fn bounded(neighbours:&mut Vec<u32>,point:usize,points:&[f32],d:usize,m:usize){if !neighbours.contains(&(point as u32)){neighbours.push(point as u32);}neighbours.sort_unstable_by(|a,b|cosine(&points[*a as usize*d..(*a as usize+1)*d],&points[point*d..(point+1)*d]).total_cmp(&cosine(&points[*b as usize*d..(*b as usize+1)*d],&points[point*d..(point+1)*d])).then(a.cmp(b)));neighbours.truncate(m);}

/// Deterministic multi-layer HNSW. Build quality is exact-neighbour based;
/// query is graph based and bounded by ef_search.
#[wasm_bindgen] pub struct HnswIndex{points:Vec<f32>,count:usize,dimension:usize,m:usize,levels:Vec<usize>,layers:Vec<Vec<Vec<u32>>>,entry:usize}
#[wasm_bindgen] impl HnswIndex{
 #[wasm_bindgen(constructor)] pub fn new(points:Vec<f32>,count:usize,dimension:usize,m:usize,seed:u32)->Result<HnswIndex,JsValue>{matrix(&points,count,dimension,"HNSW points")?;if m==0{return Err(err("HNSW m must be positive"));}let levels:(Vec<usize>)=(0..count).map(|i|level(i,seed)).collect();let max=*levels.iter().max().unwrap_or(&0);let mut layers:(Vec<Vec<Vec<u32>>>)=(0..=max).map(|_|(0..count).map(|_|Vec::new()).collect()).collect();for point in 0..count{for l in 0..=levels[point]{let mut candidates:(Vec<(f32,usize)>)=(0..point).filter(|other|levels[*other]>=l).map(|other|(cosine(&points[point*dimension..(point+1)*dimension],&points[other*dimension..(other+1)*dimension]),other)).collect();candidates.sort_unstable_by(|a,b|a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));for(_,other)in candidates.into_iter().take(m){bounded(&mut layers[l][point],other,&points,dimension,m);bounded(&mut layers[l][other],point,&points,dimension,m);}}}let entry=(0..count).max_by_key(|i|(levels[*i],std::cmp::Reverse(*i))).unwrap_or(0);Ok(HnswIndex{points,count,dimension,m,levels,layers,entry})}
 #[wasm_bindgen(getter)]pub fn count(&self)->usize{self.count} #[wasm_bindgen(getter)]pub fn dimension(&self)->usize{self.dimension} #[wasm_bindgen(getter)]pub fn max_level(&self)->usize{*self.levels.iter().max().unwrap_or(&0)}
 pub fn search(&self,query:Vec<f32>,k:usize)->Result<Vec<u32>,JsValue>{self.search_with_ef(query,k,self.m.saturating_mul(4).max(k))}
 pub fn search_with_ef(&self,query:Vec<f32>,k:usize,ef:usize)->Result<Vec<u32>,JsValue>{if query.len()!=self.dimension||query.iter().any(|x|!x.is_finite()){return Err(err("query dimension or values invalid"));}if k==0||k>self.count{return Err(err("k must be between 1 and HNSW count"));}let mut current=self.entry;for l in(1..=self.max_level()).rev(){loop{let here=cosine(&query,&self.points[current*self.dimension..(current+1)*self.dimension]);let next=self.layers[l][current].iter().map(|x|*x as usize).min_by(|a,b|cosine(&query,&self.points[*a*self.dimension..(*a+1)*self.dimension]).total_cmp(&cosine(&query,&self.points[*b*self.dimension..(*b+1)*self.dimension])).then(a.cmp(b)));if let Some(candidate)=next{if cosine(&query,&self.points[candidate*self.dimension..(candidate+1)*self.dimension])<here{current=candidate;continue;}}break;}}
 let mut seen=vec![false;self.count];let mut frontier=vec![current];let mut candidates=Vec::<(f32,usize)>::new();let cap=ef.max(k).min(self.count);while let Some(node)=frontier.pop(){if seen[node]{continue;}seen[node]=true;candidates.push((cosine(&query,&self.points[node*self.dimension..(node+1)*self.dimension]),node));for edge in &self.layers[0][node]{if !seen[*edge as usize]{frontier.push(*edge as usize);}}if candidates.len()>cap*2{candidates.sort_unstable_by(|a,b|a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));candidates.truncate(cap);}}candidates.sort_unstable_by(|a,b|a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));candidates.dedup_by_key(|x|x.1);Ok(candidates.into_iter().take(k).map(|(_,i)|i as u32).collect())}
}

struct Uf{p:Vec<usize>,rank:Vec<u8>}impl Uf{fn new(n:usize)->Self{Self{p:(0..n).collect(),rank:vec![0;n]}}fn find(&mut self,i:usize)->usize{if self.p[i]!=i{let r=self.find(self.p[i]);self.p[i]=r;}self.p[i]}fn join(&mut self,a:usize,b:usize)->bool{let(mut x,mut y)=(self.find(a),self.find(b));if x==y{return false;}if self.rank[x]<self.rank[y]{std::mem::swap(&mut x,&mut y);}self.p[y]=x;if self.rank[x]==self.rank[y]{self.rank[x]+=1;}true}}
#[derive(Serialize)]struct MstResult{edges:Vec<f32>,edge_count:usize}
#[wasm_bindgen]pub fn mst(distances:Vec<f32>,n:usize)->Result<Vec<f32>,JsValue>{square(&distances,n,"distance matrix")?;if n==1{return Ok(vec![]);}let(mut used,mut best,mut from)=(vec![false;n],vec![f32::INFINITY;n],vec![0;n]);best[0]=0.;let mut edges=Vec::with_capacity((n-1)*3);for _ in 0..n{let node=(0..n).filter(|i|!used[*i]).min_by(|a,b|best[*a].total_cmp(&best[*b]).then(a.cmp(b))).ok_or_else(||err("MST disconnected"))?;used[node]=true;if node!=0{edges.extend([from[node]as f32,node as f32,best[node]]);}for other in 0..n{let w=distances[node*n+other];if !used[other]&&w<best[other]{best[other]=w;from[other]=node;}}}Ok(edges)}

/// Sparse Kruskal MST for HDBSCAN mutual-reachability edges.
#[wasm_bindgen]pub fn mutual_reachability_mst(indices:Vec<u32>,distances:Vec<f32>,n:usize,k:usize,min_samples:usize)->Result<JsValue,JsValue>{if n==0||k==0||k>=n{return Err(err("invalid row_count or k"));}let len=product(n,k,"kNN graph")?;if indices.len()!=len||distances.len()!=len{return Err(err("kNN buffers have invalid length"));}if indices.iter().any(|x|*x as usize>=n)||distances.iter().any(|x|!x.is_finite()||*x<0.){return Err(err("kNN graph has invalid values"));}let rank=min_samples.max(1).min(k)-1;let core:(Vec<f32>)=(0..n).map(|i|{let mut local=distances[i*k..(i+1)*k].to_vec();local.sort_unstable_by(|a,b|a.total_cmp(b));local[rank]}).collect();let mut graph=Vec::<(f32,usize,usize)>::with_capacity(len);for i in 0..n{for r in 0..k{let j=indices[i*k+r]as usize;if i!=j{graph.push((core[i].max(core[j]).max(distances[i*k+r]),i.min(j),i.max(j)));}}}graph.sort_unstable_by(|a,b|a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));graph.dedup_by(|a,b|a.1==b.1&&a.2==b.2);let mut uf=Uf::new(n);let mut edges=Vec::with_capacity(n.saturating_sub(1)*3);for(w,a,b)in graph{if uf.join(a,b){edges.extend([a as f32,b as f32,w]);if edges.len()/3==n-1{break;}}}serde_wasm_bindgen::to_value(&MstResult{edge_count:edges.len()/3,edges}).map_err(|e|err(e.to_string()))}

/// Exact Euclidean HDBSCAN MST. It computes min_samples core distances in
/// tiled scans, then runs Prim over the implicit complete
/// mutual-reachability graph. Only O(n) state and n-1 output edges are kept.
#[wasm_bindgen]
pub fn euclidean_mutual_reachability_mst(rows:Vec<f32>,n:usize,d:usize,min_samples:usize,tile:usize)->Result<JsValue,JsValue>{
    matrix(&rows,n,d,"rows")?;if n<2{return Err(err("row_count must be at least 2"));}
    // hdbscan's neighbour query includes the point itself at distance zero.
    // Our tiled scan excludes self, so its min_samples-th query distance is
    // non-self rank min_samples - 2 (and is zero for min_samples=1).
    let rank=min_samples.saturating_sub(2).min(n-2);let t=tile.max(1).min(n);let mut core=vec![f32::INFINITY;n];
    for start in (0..n).step_by(t){let end=(start+t).min(n);let mut nearest:(Vec<Vec<f32>>)=(start..end).map(|_|Vec::with_capacity(rank+1+t)).collect();for target in (0..n).step_by(t){for i in start..end{for j in target..(target+t).min(n){if i!=j{nearest[i-start].push(euclidean(&rows[i*d..(i+1)*d],&rows[j*d..(j+1)*d]));}}nearest[i-start].sort_unstable_by(|a,b|a.total_cmp(b));nearest[i-start].truncate(rank+1);}}for i in start..end{core[i]=if min_samples<=1{0.0}else{nearest[i-start][rank]};}}
    let(mut used,mut best,mut from)=(vec![false;n],vec![f32::INFINITY;n],vec![0usize;n]);best[0]=0.;let mut edges=Vec::with_capacity(n.saturating_sub(1)*3);
    for _ in 0..n{let node=(0..n).filter(|i|!used[*i]).min_by(|a,b|best[*a].total_cmp(&best[*b]).then(a.cmp(b))).ok_or_else(||err("MST disconnected"))?;used[node]=true;if node!=0{edges.extend([from[node]as f32,node as f32,best[node]]);}for other in 0..n{if !used[other]{let distance=core[node].max(core[other]).max(euclidean(&rows[node*d..(node+1)*d],&rows[other*d..(other+1)*d]));if distance<best[other]||(distance==best[other]&&node<from[other]){best[other]=distance;from[other]=node;}}}}
    serde_wasm_bindgen::to_value(&MstResult{edge_count:edges.len()/3,edges}).map_err(|e|err(e.to_string()))
}

/// Result of extracting flat HDBSCAN clusters from a mutual-reachability MST.
/// `labels` uses -1 for noise.  `selection_method` is 0 for excess-of-mass
/// (EOM) and 1 for leaf selection.
#[derive(Serialize, Debug, Clone)]
struct HdbscanResult {
    labels: Vec<i32>,
    probabilities: Vec<f32>,
    outlier_scores: Vec<f32>,
    cluster_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    memberships: Option<Vec<f32>>,
}

#[derive(Clone, Copy)]
struct LinkageNode {
    left: Option<usize>,
    right: Option<usize>,
    size: usize,
    distance: f32,
}

struct LinkageUf {
    parent: Vec<usize>,
    rank: Vec<u8>,
    node: Vec<usize>,
}

impl LinkageUf {
    fn new(n: usize) -> Self {
        Self { parent: (0..n).collect(), rank: vec![0; n], node: (0..n).collect() }
    }
    fn find(&mut self, item: usize) -> usize {
        if self.parent[item] != item {
            let root = self.find(self.parent[item]);
            self.parent[item] = root;
        }
        self.parent[item]
    }
    fn component_node(&mut self, item: usize) -> usize {
        let root = self.find(item);
        self.node[root]
    }
    fn join(&mut self, left: usize, right: usize, node: usize) -> bool {
        let mut a = self.find(left);
        let mut b = self.find(right);
        if a == b { return false; }
        if self.rank[a] < self.rank[b] || (self.rank[a] == self.rank[b] && a > b) {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a;
        if self.rank[a] == self.rank[b] { self.rank[a] += 1; }
        self.node[a] = node;
        true
    }
}

struct CondensedCluster {
    node: usize,
    birth: f32,
    stability: f32,
    children: Vec<usize>,
    exits: Vec<(usize, f32)>,
    selectable: bool,
}

const MAX_LAMBDA: f32 = 1.0 / EPS;

fn lambda(distance: f32) -> f32 {
    if distance <= EPS { MAX_LAMBDA } else { (1.0 / distance).min(MAX_LAMBDA) }
}

fn leaves_under(nodes: &[LinkageNode], root: usize, output: &mut Vec<usize>) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if let (Some(left), Some(right)) = (nodes[node].left, nodes[node].right) {
            // Push right first so point iteration is deterministic ascending by
            // the linkage's left/right construction order.
            stack.push(right);
            stack.push(left);
        } else {
            output.push(node);
        }
    }
}

fn record_exit(
    nodes: &[LinkageNode],
    root: usize,
    cluster: usize,
    at_lambda: f32,
    clusters: &mut [CondensedCluster],
) {
    let size = nodes[root].size;
    clusters[cluster].stability += size as f32 * (at_lambda - clusters[cluster].birth).max(0.0);
    let mut leaves = Vec::with_capacity(size);
    leaves_under(nodes, root, &mut leaves);
    clusters[cluster].exits.extend(leaves.into_iter().map(|point| (point, at_lambda)));
}

fn condense_cluster(
    nodes: &[LinkageNode],
    node: usize,
    cluster: usize,
    min_cluster_size: usize,
    clusters: &mut Vec<CondensedCluster>,
) {
    let (left, right) = match (nodes[node].left, nodes[node].right) {
        (Some(left), Some(right)) => (left, right),
        _ => return,
    };
    let split_lambda = lambda(nodes[node].distance);
    let left_big = nodes[left].size >= min_cluster_size;
    let right_big = nodes[right].size >= min_cluster_size;
    match (left_big, right_big) {
        (true, true) => {
            record_exit(nodes, left, cluster, split_lambda, clusters);
            record_exit(nodes, right, cluster, split_lambda, clusters);
            let left_cluster = clusters.len();
            clusters.push(CondensedCluster { node: left, birth: split_lambda, stability: 0.0, children: Vec::new(), exits: Vec::new(), selectable: true });
            let right_cluster = clusters.len();
            clusters.push(CondensedCluster { node: right, birth: split_lambda, stability: 0.0, children: Vec::new(), exits: Vec::new(), selectable: true });
            clusters[cluster].children.extend([left_cluster, right_cluster]);
            condense_cluster(nodes, left, left_cluster, min_cluster_size, clusters);
            condense_cluster(nodes, right, right_cluster, min_cluster_size, clusters);
        }
        (true, false) => {
            record_exit(nodes, right, cluster, split_lambda, clusters);
            condense_cluster(nodes, left, cluster, min_cluster_size, clusters);
        }
        (false, true) => {
            record_exit(nodes, left, cluster, split_lambda, clusters);
            condense_cluster(nodes, right, cluster, min_cluster_size, clusters);
        }
        (false, false) => {
            record_exit(nodes, left, cluster, split_lambda, clusters);
            record_exit(nodes, right, cluster, split_lambda, clusters);
        }
    }
}

fn select_eom(cluster: usize, clusters: &[CondensedCluster], selected: &mut Vec<usize>) -> f32 {
    // Each recursive child appends its selected descendants. Keeping the
    // insertion point lets an EOM parent replace that entire descendant set
    // in O(number of selected descendants), with no ancestor scans.
    let selected_start = selected.len();
    let mut child_stability = 0.0;
    for child in &clusters[cluster].children {
        child_stability += select_eom(*child, clusters, selected);
    }
    if clusters[cluster].selectable && clusters[cluster].stability >= child_stability {
        selected.truncate(selected_start);
        selected.push(cluster);
        clusters[cluster].stability
    } else {
        child_stability
    }
}

fn select_leaves(cluster: usize, clusters: &[CondensedCluster], selected: &mut Vec<usize>) {
    if clusters[cluster].children.is_empty() {
        if clusters[cluster].selectable { selected.push(cluster); }
    } else {
        for child in &clusters[cluster].children { select_leaves(*child, clusters, selected); }
    }
}

fn hdbscan_extract_impl(
    edges: &[f32], row_count: usize, min_cluster_size: usize, selection_method: u32,
    allow_single_cluster: bool,
    rows: Option<&[f32]>,
    dimension: usize,
) -> Result<HdbscanResult, String> {
    if row_count == 0 { return Err("row_count must be positive".into()); }
    if min_cluster_size < 2 { return Err("min_cluster_size must be at least 2".into()); }
    if selection_method > 1 { return Err("selection_method must be 0 (EOM) or 1 (leaf)".into()); }
    if edges.len() % 3 != 0 { return Err("MST edges must be endpoint, endpoint, distance triples".into()); }
    let mut sorted = Vec::with_capacity(edges.len() / 3);
    for (index, edge) in edges.chunks_exact(3).enumerate() {
        let (left, right, distance) = (edge[0], edge[1], edge[2]);
        if !left.is_finite() || !right.is_finite() || !distance.is_finite() || distance < 0.0 {
            return Err("MST edges must be finite with non-negative distances".into());
        }
        if left.fract() != 0.0 || right.fract() != 0.0 || left < 0.0 || right < 0.0 || left as usize >= row_count || right as usize >= row_count || left == right {
            return Err("MST edge endpoints must be distinct valid integer row indices".into());
        }
        let a = left as usize;
        let b = right as usize;
        sorted.push((distance, a.min(b), a.max(b), index));
    }
    sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)).then(a.3.cmp(&b.3)));
    let mut nodes = (0..row_count).map(|_| LinkageNode { left: None, right: None, size: 1, distance: 0.0 }).collect::<Vec<_>>();
    let mut uf = LinkageUf::new(row_count);
    for (distance, left, right, _) in sorted {
        let left_node = uf.component_node(left);
        let right_node = uf.component_node(right);
        if left_node == right_node { continue; }
        let parent = nodes.len();
        nodes.push(LinkageNode { left: Some(left_node), right: Some(right_node), size: nodes[left_node].size + nodes[right_node].size, distance });
        uf.join(left, right, parent);
    }
    if nodes.len().saturating_sub(row_count) != row_count.saturating_sub(1) {
        return Err("MST must contain row_count - 1 acyclic edges and connect every row".into());
    }
    let mut roots = Vec::new();
    for point in 0..row_count {
        let root = uf.component_node(point);
        if !roots.contains(&root) { roots.push(root); }
    }
    roots.sort_unstable();
    let mut clusters = Vec::new();
    let mut root_clusters = Vec::new();
    for root in roots {
        if nodes[root].size < min_cluster_size { continue; }
        let cluster = clusters.len();
        clusters.push(CondensedCluster { node: root, birth: 0.0, stability: 0.0, children: Vec::new(), exits: Vec::new(), selectable: allow_single_cluster });
        root_clusters.push(cluster);
        condense_cluster(&nodes, root, cluster, min_cluster_size, &mut clusters);
    }
    let mut selected = Vec::new();
    for &root in &root_clusters {
        if selection_method == 0 { select_eom(root, &clusters, &mut selected); }
        else { select_leaves(root, &clusters, &mut selected); }
    }
    selected.sort_unstable();
    selected.dedup();
    let mut labels = vec![-1; row_count];
    let mut probabilities = vec![0.0; row_count];
    for (label, cluster) in selected.iter().enumerate() {
        let max_exit = clusters[*cluster].exits.iter().map(|(_, value)| *value).fold(0.0_f32, f32::max);
        for (point, exit_lambda) in &clusters[*cluster].exits {
            labels[*point] = label as i32;
            probabilities[*point] = if max_exit > 0.0 { (exit_lambda / max_exit).clamp(0.0, 1.0) } else { 1.0 };
        }
    }
    let outlier_scores = probabilities.iter().map(|probability| 1.0 - probability).collect();
    let memberships = rows.map(|values| all_points_memberships(&nodes, &clusters, &root_clusters, &selected, row_count, min_cluster_size, values, dimension));
    Ok(HdbscanResult { labels, probabilities, outlier_scores, cluster_count: selected.len(), memberships })
}

/// Build the raw condensed-tree information needed by hdbscan's prediction
/// module.  The linkage tree is binary, while a condensed tree replaces a
/// child smaller than min_cluster_size with one row per point.
fn condensed_prediction_data(
    nodes: &[LinkageNode], clusters: &[CondensedCluster], roots: &[usize], min_size: usize,
) -> (Vec<Option<(usize, f32)>>, Vec<Option<(usize, f32)>>, Vec<f32>, Vec<Vec<usize>>) {
    let mut parent = vec![None; clusters.len()];
    let mut point_parent = vec![None; nodes.iter().filter(|n| n.left.is_none()).count()];
    let mut point_rows: Vec<Vec<(usize, f32)>> = (0..clusters.len()).map(|_| Vec::new()).collect();
    let mut max_lambda = vec![0.0; clusters.len()];
    let mut leaves: Vec<Vec<usize>> = (0..clusters.len()).map(|_| Vec::new()).collect();
    fn visit(nodes: &[LinkageNode], clusters: &[CondensedCluster], node: usize, cluster: usize, min_size: usize,
             parent: &mut [Option<(usize, f32)>], point_parent: &mut [Option<(usize, f32)>],
             point_rows: &mut [Vec<(usize, f32)>], max_lambda: &mut [f32]) {
        let (left, right) = match (nodes[node].left, nodes[node].right) { (Some(left), Some(right)) => (left, right), _ => return };
        let at = lambda(nodes[node].distance);
        max_lambda[cluster] = max_lambda[cluster].max(at);
        for child in [left, right] {
            if nodes[child].size >= min_size {
                let child_cluster = clusters[cluster].children.iter().find(|c| clusters[**c].node == child).copied();
                // A one-big/one-small condensation split retains the same
                // condensed cluster id for the big branch; only a
                // two-big split creates a child cluster id.
                let cc = child_cluster.unwrap_or(cluster);
                if cc != cluster { parent[cc] = Some((cluster, at)); }
                visit(nodes, clusters, child, cc, min_size, parent, point_parent, point_rows, max_lambda);
            } else {
                let mut points = Vec::new(); leaves_under(nodes, child, &mut points);
                for point in points {
                    point_parent[point] = Some((cluster, at));
                    point_rows[cluster].push((point, at));
                    max_lambda[cluster] = max_lambda[cluster].max(at);
                }
            }
        }
    }
    for &root in roots {
        let cluster = root;
        visit(nodes, clusters, clusters[cluster].node, cluster, min_size, &mut parent, &mut point_parent, &mut point_rows, &mut max_lambda);
    }
    fn leaf_dfs(cluster: usize, clusters: &[CondensedCluster], leaves: &mut Vec<usize>) {
        if clusters[cluster].children.is_empty() { leaves.push(cluster); }
        else { for &child in &clusters[cluster].children { leaf_dfs(child, clusters, leaves); } }
    }
    for cluster in 0..clusters.len() { leaf_dfs(cluster, clusters, &mut leaves[cluster]); }
    let exemplar_lists = leaves.iter().map(|leaf_clusters| {
        let mut out = Vec::new();
        for &leaf in leaf_clusters {
            let max = point_rows[leaf].iter().map(|(_, l)| *l).fold(0.0, f32::max);
            out.extend(point_rows[leaf].iter().filter(|(_, l)| *l == max).map(|(p, _)| *p));
        }
        // The Python implementation preserves raw-tree row order. Linkage
        // traversal is deterministic, so this is already stable; sorting is
        // only a tie-breaker for equivalent edge orderings.
        out
    }).collect();
    (parent, point_parent, max_lambda, exemplar_lists)
}

fn merge_height(point_cluster: usize, point_lambda: f32, selected: usize, parent: &[Option<(usize, f32)>]) -> f64 {
    let mut left = point_cluster; let mut right = selected; let mut took_left = false; let mut took_right = false; let mut last = point_lambda;
    let mut guard = 0;
    while left != right && guard <= parent.len() { guard += 1;
        if left > right {
            took_left = true; last = parent[left].map(|(_, l)| l).unwrap_or(point_lambda); left = parent[left].map(|(p, _)| p).unwrap_or(left);
        } else {
            took_right = true; last = parent[right].map(|(_, l)| l).unwrap_or(point_lambda); right = parent[right].map(|(p, _)| p).unwrap_or(right);
        }
    }
    if took_left && took_right { last as f64 } else { point_lambda as f64 }
}

fn all_points_memberships(nodes: &[LinkageNode], clusters: &[CondensedCluster], roots: &[usize], selected: &[usize], count: usize, min_size: usize, rows: &[f32], dimension: usize) -> Vec<f32> {
    if selected.is_empty() { return Vec::new(); }
    let (parent, point_parent, leaf_max, exemplar_lists) = condensed_prediction_data(nodes, clusters, roots, min_size);
    let c = selected.len();
    let mut exemplars: Vec<Vec<usize>> = Vec::with_capacity(c);
    for &cluster in selected { exemplars.push(exemplar_lists[cluster].clone()); }
    let mut result = vec![0.0f32; count * c];
    for point in 0..count {
        let (point_cluster, point_lambda) = match point_parent[point] { Some(v) => v, None => continue };
        let mut heights = vec![0.0f64; c];
        let mut dist = vec![0.0f64; c];
        let row = &rows[point * dimension..(point + 1) * dimension];
        for (j, &cluster) in selected.iter().enumerate() {
            heights[j] = merge_height(point_cluster, point_lambda, cluster, &parent);
            let mut best = f64::MAX;
            for &exemplar in &exemplars[j] {
                let other = &rows[exemplar * dimension..(exemplar + 1) * dimension];
                let d = row.iter().zip(other).map(|(a,b)| { let x = *a as f64 - *b as f64; x*x }).sum::<f64>().sqrt();
                best = best.min(d);
            }
            dist[j] = if best > 0.0 { 1.0 / best } else { f64::MAX / c as f64 };
        }
        let dist_sum = dist.iter().sum::<f64>();
        let per_cluster_scores = heights.iter().map(|&height| {
            let max = leaf_max.get(point_cluster).copied().unwrap_or(point_lambda) + 1e-8;
            if height > 0.0 { (-(max as f64) / height).exp() } else { 0.0 }
        }).collect::<Vec<_>>();
        let score_max = per_cluster_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let outlier = per_cluster_scores.iter().map(|score| (score - score_max).exp()).collect::<Vec<_>>();
        let outlier_sum = outlier.iter().sum::<f64>();
        let max_height_index = heights.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).map(|(i,_)| i).unwrap_or(0);
        let in_prob = heights.iter().copied().fold(0.0, f64::max) / leaf_max.get(selected[max_height_index]).copied().unwrap_or(point_lambda).max(point_lambda).max(1e-8) as f64;
        for j in 0..c {
            // all_points_membership_vectors: distance_vec * outlier_vec,
            // normalize rows, then scale by probability in some cluster.
            result[point*c+j] = ((dist[j] / dist_sum) * (outlier[j] / outlier_sum) * in_prob) as f32;
        }
    }
    result
}

/// Extract HDBSCAN clusters from mutual-reachability MST triples.
///
/// This performs a full single-linkage hierarchy build, condensation at
/// `min_cluster_size`, stability calculation, and deterministic EOM or leaf
/// selection. It deliberately does not apply a percentile distance cut.
#[wasm_bindgen]
pub fn hdbscan_extract(
    mst_edges: Vec<f32>,
    row_count: usize,
    min_cluster_size: usize,
    selection_method: u32,
    allow_single_cluster: bool,
) -> Result<JsValue, JsValue> {
    let result = hdbscan_extract_impl(&mst_edges, row_count, min_cluster_size, selection_method, allow_single_cluster, None, 0).map_err(err)?;
    serde_wasm_bindgen::to_value(&result).map_err(|error| err(error.to_string()))
}

/// Extraction variant that also computes Python-compatible all-points soft
/// memberships from the original feature rows. Keeping this as a separate
/// export preserves the compact legacy extraction ABI for callers that only
/// need labels and probabilities.
#[wasm_bindgen]
pub fn hdbscan_extract_with_rows(
    mst_edges: Vec<f32>, rows: Vec<f32>, row_count: usize, dimension: usize,
    min_cluster_size: usize, selection_method: u32, allow_single_cluster: bool,
) -> Result<JsValue, JsValue> {
    matrix(&rows, row_count, dimension, "rows").map_err(|value| value)?;
    let result = hdbscan_extract_impl(&mst_edges, row_count, min_cluster_size, selection_method, allow_single_cluster, Some(&rows), dimension).map_err(err)?;
    serde_wasm_bindgen::to_value(&result).map_err(|error| err(error.to_string()))
}

#[cfg(test)]mod tests{use super::*;
 #[test]fn matrix_math(){assert_eq!(matmul(vec![1.,2.,3.,4.],vec![2.,0.,1.,2.],2,2,2).unwrap(),vec![4.,4.,10.,8.]);}
 #[test]fn cosine_knn_excludes_self(){let distances=cosine_distances_tiled(vec![1.,0.,0.9,0.1,0.,1.],3,2,2).unwrap();assert_eq!(exact_knn(distances,3,1).unwrap(),vec![1,0,1]);}
 #[test]fn hnsw_returns_k(){let h=HnswIndex::new(vec![1.,0.,0.9,0.1,0.,1.],3,2,2,42).unwrap();assert_eq!(h.search(vec![1.,0.],2).unwrap().len(),2);}
 #[test]fn mst_shapes(){assert_eq!(mst(vec![0.,1.,2.,1.,0.,1.,2.,1.,0.],3).unwrap().len(),6);}
 #[test]
 fn hdbscan_extracts_separated_dense_components_deterministically(){
    let edges=vec![0.,1.,0.05,1.,2.,0.1,3.,4.,0.05,4.,5.,0.1,2.,3.,10.];
    let forward=hdbscan_extract_impl(&edges,6,3,0,false,None,0).unwrap();
    let reversed=vec![2.,3.,10.,4.,5.,0.1,3.,4.,0.05,1.,2.,0.1,0.,1.,0.05];
    let backward=hdbscan_extract_impl(&reversed,6,3,0,false,None,0).unwrap();
    assert_eq!(forward.labels,vec![0,0,0,1,1,1]);
    assert_eq!(forward.labels,backward.labels);
    assert_eq!(forward.cluster_count,2);
    assert!(forward.probabilities.iter().all(|value|*value>0.0&&*value<=1.0));
 }
 #[test]
 fn hdbscan_eom_prefers_stable_parent_while_leaf_returns_children(){
    // Two four-point groups each contain two very short-lived two-point
    // children. EOM retains the stable four-point parents; leaf selection
    // deliberately returns the four terminal condensed-tree clusters.
    let edges=vec![0.,1.,0.99,2.,3.,0.99,4.,5.,0.99,6.,7.,0.99,1.,2.,1.,5.,6.,1.,3.,4.,10.];
    let eom=hdbscan_extract_impl(&edges,8,2,0,false,None,0).unwrap();
    let leaf=hdbscan_extract_impl(&edges,8,2,1,false,None,0).unwrap();
    assert_eq!(eom.cluster_count,2);
    assert_eq!(eom.labels,vec![0,0,0,0,1,1,1,1]);
    assert_eq!(leaf.cluster_count,4);
    assert_eq!(leaf.labels,vec![0,0,1,1,2,2,3,3]);
 }
 #[test]
 fn hdbscan_marks_root_only_and_too_small_cases_as_noise_by_default(){
    let root_only=hdbscan_extract_impl(&[0.,1.,0.1,1.,2.,0.1],3,3,0,false,None,0).unwrap();
    assert_eq!(root_only.labels,vec![-1,-1,-1]);
    let permitted=hdbscan_extract_impl(&[0.,1.,0.1,1.,2.,0.1],3,3,0,true,None,0).unwrap();
    assert_eq!(permitted.labels,vec![0,0,0]);
    let too_small=hdbscan_extract_impl(&[0.,1.,0.1,1.,2.,0.1],3,4,0,true,None,0).unwrap();
    assert_eq!(too_small.labels,vec![-1,-1,-1]);
    assert!(too_small.outlier_scores.iter().all(|value|*value==1.0));
 }
 #[test]
 fn hdbscan_probabilities_reflect_point_exit_lambda(){
    // In each selected four-point cluster, one point leaves at lambda=1 and
    // the other three leave at lambda=2, so their persistence probabilities
    // are respectively 0.5 and 1.0.
    let edges=vec![0.,1.,0.5,1.,2.,0.5,2.,3.,1.,4.,5.,0.5,5.,6.,0.5,6.,7.,1.,3.,4.,10.];
    let result=hdbscan_extract_impl(&edges,8,3,0,false,None,0).unwrap();
    assert_eq!(result.labels,vec![0,0,0,0,1,1,1,1]);
    assert_eq!(result.probabilities,vec![1.,1.,1.,0.5,1.,1.,1.,0.5]);
    assert_eq!(result.outlier_scores,vec![0.,0.,0.,0.5,0.,0.,0.,0.5]);
 }
 #[test]
 fn hdbscan_rejects_an_incomplete_mst(){
    let error=hdbscan_extract_impl(&[0.,1.,0.1],3,2,0,false,None,0).unwrap_err();
    assert!(error.contains("connect every row"));
 }
 #[test]
 fn all_points_memberships_are_soft_and_row_scaled(){
    let edges=vec![0.,1.,0.05,1.,2.,0.1,3.,4.,0.05,4.,5.,0.1,2.,3.,10.];
    let rows=vec![1.,0.,0.995,0.1,0.98,0.2,-1.,0.,-0.995,-0.1,-0.98,-0.2];
    let result=hdbscan_extract_impl(&edges,6,3,0,false,Some(&rows),2).unwrap();
    let memberships=result.memberships.unwrap();
    assert_eq!(memberships.len(),12);
    for point in 0..6 { let sum: f32=memberships[point*2..point*2+2].iter().sum(); assert!(sum <= 1.00001); }
    assert!(memberships[0] > 0.5 && memberships[1] < 0.5);
    assert!(memberships[3] < 0.5 && memberships[4] > 0.5);
 }
 #[test]
 fn all_points_noise_keeps_nonzero_soft_mass(){
    // Two selected three-point clusters, with a two-point discarded branch
    // nested below the first cluster. The discarded points are noise labels,
    // but prediction membership remains nonzero after scaling.
    let edges=vec![0.,1.,0.5,1.,2.,0.5,6.,7.,0.5,2.,6.,2.,3.,4.,0.5,4.,5.,0.5,2.,3.,2.];
    let rows=vec![0.,0.,0.1,0.,0.2,0.,1.,0.,1.1,0.,1.2,0.,0.3,0.,0.4,0.];
    let result=hdbscan_extract_impl(&edges,8,3,0,false,Some(&rows),2).unwrap();
    assert_eq!(result.labels[6], -1);
    let memberships=result.memberships.unwrap();
    assert!(memberships[12] > 0.0 && memberships[12] < 1.0);
 }
}
