"""
Segundo paso del maniquí v4.1 (después de construir_cabeza_masculina.py):
ojos mejorados y sin pestañas, a petición de Pedro.

    python tools/construir_cabeza_masculina.py /tmp/mh /tmp/head_v3_neutral.glb /tmp/head_v4.glb
    python tools/mejorar_ojos_cabeza.py /tmp/head_v4.glb frontend/assets/head.glb

1. Quita la malla de pestañas de la escena (se veían como una banda negra
   arriba y gris abajo).
2. Iris marrón natural (la textura de MakeHuman lo trae rojizo) y
   esclerótica algo más limpia; material del ojo sin brillo metálico.
3. En lugar de pestañas, una línea suave más oscura en el borde de los
   párpados (color de vértice), calculada lanzando rayos de frente para
   encontrar el contorno de la abertura del ojo.
4. Ojo más liso: se subdivide dos veces y los vértices se acercan a la
   esfera ajustada de cada ojo (antes se notaban las facetas).
"""
import io
import sys

import numpy as np
from PIL import Image
from pygltflib import GLTF2, BufferView, Accessor
CT={5126:np.float32,5125:np.uint32,5121:np.uint8,5123:np.uint16}
NC={"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}
def load(path):
    g=GLTF2().load(path); blob=g.binary_blob()
    return g, blob
def read_acc(g, blob, i):
    a=g.accessors[i]; bv=g.bufferViews[a.bufferView]
    n=NC[a.type]; dt=CT[a.componentType]
    off=(bv.byteOffset or 0)+(a.byteOffset or 0)
    return np.frombuffer(blob, dtype=dt, count=a.count*n, offset=off).reshape(a.count, n) if n>1 else np.frombuffer(blob, dtype=dt, count=a.count, offset=off).copy()
def rebuild(g, blob, replace, add=()):
    """replace: {accessor_idx: array}; add: list of (array, type, componentType, normalized) -> returns new accessor indices."""
    views=[]
    for i,bv in enumerate(g.bufferViews):
        views.append(bytearray(blob[(bv.byteOffset or 0):(bv.byteOffset or 0)+bv.byteLength]))
    for ai,arr in replace.items():
        a=g.accessors[ai]; views[a.bufferView]=bytearray(np.ascontiguousarray(arr.astype(CT[a.componentType])).tobytes())
        if a.type=="VEC3" and a.componentType==5126:
            a.max=arr.max(0).tolist(); a.min=arr.min(0).tolist()
    new_ids=[]
    for arr,typ,ct,norm in add:
        views.append(bytearray(np.ascontiguousarray(arr.astype(CT[ct])).tobytes()))
        g.bufferViews.append(BufferView(buffer=0, byteOffset=0, byteLength=0))
        acc=Accessor(bufferView=len(g.bufferViews)-1, componentType=ct, count=len(arr), type=typ, normalized=norm or None)
        if typ=="SCALAR" and ct==5126: acc.max=[float(arr.max())]; acc.min=[float(arr.min())]
        g.accessors.append(acc); new_ids.append(len(g.accessors)-1)
    out=bytearray()
    for i,v in enumerate(views):
        while len(out)%4: out.append(0)
        g.bufferViews[i].byteOffset=len(out); g.bufferViews[i].byteLength=len(v); out+=v
    while len(out)%4: out.append(0)
    g.buffers[0].byteLength=len(out)
    g.set_binary_blob(bytes(out))
    return new_ids


from PIL import Image
import io, colorsys
import numpy as np
SRC, DST = sys.argv[1], sys.argv[2]
g,blob=load(SRC)
meshes={m.name:m.primitives[0] for m in g.meshes}
# 1. sin pestañas
lash_node=[i for i,n in enumerate(g.nodes) if n.name=="eyelashes"][0]
g.scenes[0].nodes=[i for i in g.scenes[0].nodes if i!=lash_node]
# 2. iris marrón natural
bv=g.bufferViews[g.images[0].bufferView]
im=Image.open(io.BytesIO(blob[bv.byteOffset:bv.byteOffset+bv.byteLength])).convert("RGBA")
a=np.asarray(im).astype(float)/255
H,W=a.shape[:2]; yy,xx=np.mgrid[0:H,0:W]
rgb=a[...,:3]
hsv=np.asarray(Image.fromarray((rgb*255).astype(np.uint8)).convert("HSV")).astype(float)/255
for cx,cy in ((722,305),(297,727)):
    d=np.hypot(xx-cx,yy-cy)
    m=np.clip((125-d)/10,0,1)      # iris (con borde suave)
    hsv[...,0]=hsv[...,0]*(1-m)+ (24/360)*m
    hsv[...,1]=hsv[...,1]*(1-m*0.35)
    hsv[...,2]=np.minimum(1,hsv[...,2]*(1+m*0.45))   # la web no corrige gamma: se ve más oscuro que aquí
    sc=np.clip((300-d)/30,0,1)*(1-m)  # esclerótica algo más limpia
    hsv[...,1]*=1-0.35*sc
    hsv[...,2]=np.minimum(1,hsv[...,2]*(1+0.40*sc))
out=np.asarray(Image.fromarray((hsv*255).astype(np.uint8),"HSV").convert("RGB")).astype(float)/255
# la pupila sigue negra: donde era muy oscura, conservar
dark=rgb.max(-1)<0.12
out[dark]=rgb[dark]
res=Image.fromarray((np.concatenate([out,a[...,3:]],-1)*255).astype(np.uint8),"RGBA").convert("RGB")
buf=io.BytesIO(); res.save(buf,"JPEG",quality=90); eye_bytes=buf.getvalue()
# 3. sombra de párpado alrededor de los ojos (sustituye a la línea de pestañas)
S=read_acc(g,blob,meshes["geometry_0"].attributes.POSITION).astype(float)
col=read_acc(g,blob,meshes["geometry_0"].attributes.COLOR_0).astype(float)
E=read_acc(g,blob,meshes["geometry_2"].attributes.POSITION).astype(float)
import trimesh
from scipy.spatial import cKDTree
F=read_acc(g,blob,meshes["geometry_0"].indices).reshape(-1,3)
FE=read_acc(g,blob,meshes["geometry_2"].indices).reshape(-1,3)
both=trimesh.Trimesh(np.vstack([S,E]),np.vstack([F,FE+len(S)]),process=False)
n_skin_faces=len(F)
# Contorno de la abertura de cada ojo visto de frente: rayos hacia -z; donde
# pasa de "da en el ojo" a "da en la piel" está el borde del párpado.
rim_pts=[]
for side in (E[:,0]>0, E[:,0]<=0):
    lo,hi=E[side].min(0),E[side].max(0)
    xs=np.arange(lo[0]-0.01,hi[0]+0.01,0.0015); ys=np.arange(lo[1]-0.01,hi[1]+0.01,0.0015)
    X,Y=np.meshgrid(xs,ys)
    orig=np.stack([X.ravel(),Y.ravel(),np.full(X.size,1.0)],1)
    locs,ray_idx,tri_idx=both.ray.intersects_location(orig,np.tile([0,0,-1.0],(len(orig),1)),multiple_hits=False)
    hit_eye=np.zeros(len(orig),bool); hit=np.zeros(len(orig),bool); z=np.full(len(orig),-9.0)
    hit[ray_idx]=True; hit_eye[ray_idx]=tri_idx>=n_skin_faces; z[ray_idx]=locs[:,2]
    HE=hit_eye.reshape(X.shape); Z=z.reshape(X.shape); Hh=hit.reshape(X.shape)
    edge=np.zeros_like(HE)
    for dy,dx in ((0,1),(1,0),(0,-1),(-1,0)):
        edge|=HE & ~np.roll(np.roll(HE,dy,0),dx,1) & np.roll(np.roll(Hh,dy,0),dx,1)
    pts=np.stack([X[edge],Y[edge],Z[edge]],1)
    rim_pts.append(pts); print("borde", len(pts))
rim=np.vstack(rim_pts)
dist,_=cKDTree(rim).query(S)
cy=np.array([E[E[:,0]>0].mean(0)[1],E[E[:,0]<=0].mean(0)[1]]).mean()
up=np.clip((S[:,1]-cy)/0.01+0.5,0,1)
w=0.018
line=np.clip(1-dist/w,0,1)**1.3
dark=np.clip(line*(0.38+0.30*up),0,0.62)
col[:,:3]*=(1-dark)[:,None]
# 4. Ojo más liso: subdividir 2 veces y llevar los vértices nuevos a la
#    esfera ajustada de cada ojo (se veían las facetas).
eprim=meshes["geometry_2"]
EUV=read_acc(g,blob,eprim.attributes.TEXCOORD_0).astype(float)
EV,EF,EU=E.copy(),FE.copy(),EUV.copy()
for _ in range(2):
    EV,EF,attr=trimesh.remesh.subdivide(EV,EF,vertex_attributes={"uv":EU})
    EU=attr["uv"]
for side in (E[:,0]>0, E[:,0]<=0):
    P0=E[side]
    A=np.c_[2*P0,np.ones(len(P0))]; bb=(P0**2).sum(1)
    sol=np.linalg.lstsq(A,bb,rcond=None)[0]; cc=sol[:3]; rr=np.sqrt(sol[3]+cc@cc)
    sel=(EV[:,0]>0) if side[0]==(E[0,0]>0) and E[0,0]>0 else (EV[:,0]<=0)
    sel=(EV[:,0]>0) if P0[:,0].mean()>0 else (EV[:,0]<=0)
    v=EV[sel]-cc; dist=np.linalg.norm(v,axis=1,keepdims=True)
    # solo acercar a la esfera (la parte del iris es algo más plana: se respeta)
    EV[sel]=cc+v/dist*(0.5*dist+0.5*rr)
    print("esfera ojo", cc.round(3), round(rr,4))
eye_mat=g.materials[meshes["geometry_2"].material]
eye_mat.pbrMetallicRoughness.metallicFactor=0.0
eye_mat.pbrMetallicRoughness.roughnessFactor=0.22
# sustituir la imagen
views=[bytearray(blob[(b.byteOffset or 0):(b.byteOffset or 0)+b.byteLength]) for b in g.bufferViews]
views[g.images[0].bufferView]=bytearray(eye_bytes); g.images[0].mimeType="image/jpeg"
g.set_binary_blob(b"".join([]))
tmp=bytearray()
for i,v in enumerate(views):
    while len(tmp)%4: tmp.append(0)
    g.bufferViews[i].byteOffset=len(tmp); g.bufferViews[i].byteLength=len(v); tmp+=v
g.buffers[0].byteLength=len(tmp); g.set_binary_blob(bytes(tmp))
ids=rebuild(g,g.binary_blob(),{meshes["geometry_0"].attributes.COLOR_0:col.round()},
            [(EV.astype(np.float32),"VEC3",5126,False),(EU.astype(np.float32),"VEC2",5126,False),(EF.reshape(-1).astype(np.uint32),"SCALAR",5125,False)])
eprim.attributes.POSITION, eprim.attributes.TEXCOORD_0, eprim.indices = ids
g.accessors[ids[0]].max=EV.max(0).tolist(); g.accessors[ids[0]].min=EV.min(0).tolist()
g.save(DST); print("ok")
