/** Tree SHAP path algorithm adapted from SHAP's MIT-licensed pytree.py.
 * Copyright (c) 2018 Scott Lundberg. Full notice in THIRD_PARTY_NOTICES.md.
 * LIME and anchor-style browser explainers are separate educational adaptations.
 */
export function predict(model, input) {
  const x=input.map(Math.fround);let sum=0;
  for(const t of model.trees){let n=0;while(t.left[n]!==-1)n=x[t.feature[n]]<=t.threshold[n]?t.left[n]:t.right[n];sum+=t.value[n];}
  return sum/model.trees.length;
}
function extend(fi,z,o,p,d,zf,of,f){fi[d]=f;z[d]=zf;o[d]=of;p[d]=d===0?1:0;for(let i=d-1;i>=0;i--){p[i+1]+=of*p[i]*(i+1)/(d+1);p[i]=zf*p[i]*(d-i)/(d+1);}}
function unwind(fi,z,o,p,d,k){let next=p[d];for(let i=d-1;i>=0;i--){if(o[k]!==0){const tmp=p[i];p[i]=next*(d+1)/((i+1)*o[k]);next=tmp-p[i]*z[k]*(d-i)/(d+1);}else p[i]=p[i]*(d+1)/(z[k]*(d-i));}for(let i=k;i<d;i++){fi[i]=fi[i+1];z[i]=z[i+1];o[i]=o[i+1];}}
function unwound(z,o,p,d,k){let next=p[d],total=0;for(let i=d-1;i>=0;i--){if(o[k]!==0){let tmp=next*(d+1)/((i+1)*o[k]);total+=tmp;next=p[i]-tmp*z[k]*(d-i)/(d+1);}else total+=(p[i]/z[k])/((d-i)/(d+1));}return total;}
export function treeShap(model,input){
 const x=input.map(Math.fround),phi=Array(input.length).fill(0);
 for(const t of model.trees){
  function rec(n,d,fi0,z0,o0,p0,zf,of,feature){
   const fi=fi0.slice(),z=z0.slice(),o=o0.slice(),p=p0.slice();extend(fi,z,o,p,d,zf,of,feature);
   if(t.left[n]===-1){for(let i=1;i<=d;i++)phi[fi[i]]+=unwound(z,o,p,d,i)*(o[i]-z[i])*t.value[n];return;}
   const f=t.feature[n],hot=x[f]<=t.threshold[n]?t.left[n]:t.right[n],cold=hot===t.left[n]?t.right[n]:t.left[n];
   const hz=t.cover[hot]/t.cover[n],cz=t.cover[cold]/t.cover[n];let iz=1,io=1,k=0;
   while(k<=d&&fi[k]!==f)k++;
   if(k!==d+1){iz=z[k];io=o[k];unwind(fi,z,o,p,d,k);d--;}
   rec(hot,d+1,fi,z,o,p,hz*iz,io,f);rec(cold,d+1,fi,z,o,p,cz*iz,0,f);
  }
  rec(0,0,[],[],[],[],1,1,-1);
 }
 return {base:model.base,values:phi.map(v=>v/model.trees.length)};
}
export function seeded(seed=42){return ()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296;};}
const bin=(v,q)=>q.reduce((n,b)=>n+(v>b),0);
function solve(A,b){const a=A.map((r,i)=>[...r,b[i]]),n=b.length;for(let i=0;i<n;i++){let k=i;for(let j=i+1;j<n;j++)if(Math.abs(a[j][i])>Math.abs(a[k][i]))k=j;[a[i],a[k]]=[a[k],a[i]];let d=a[i][i];if(Math.abs(d)<1e-12)d=1e-12;for(let j=i;j<=n;j++)a[i][j]/=d;for(let r=0;r<n;r++)if(r!==i){let m=a[r][i];for(let j=i;j<=n;j++)a[r][j]-=m*a[i][j];}}return a.map(r=>r[n]);}
export function browserLime(model,x,samples=768){
 const rng=seeded(42),d=x.length,xb=x.map((v,i)=>bin(v,model.quantiles[i]));const Z=[],Y=[],W=[];
 for(let n=0;n<samples;n++){const row=n===0?x:model.features.map((_,j)=>model.train[Math.floor(rng()*model.train.length)][j]);let z=row.map((v,j)=>+(bin(v,model.quantiles[j])===xb[j]));Z.push(z);Y.push(predict(model,row));let dist=z.reduce((s,v)=>s+(1-v),0);W.push(Math.exp(-dist/(.75*.75*d)));}
 const mean=Y.reduce((s,y,i)=>s+y*W[i],0)/W.reduce((a,b)=>a+b,0);
 // Weighted ridge on all bin indicators; then refit the three largest weights.
 function fit(indices){let size=indices.length+1,A=Array.from({length:size},()=>Array(size).fill(0)),b=Array(size).fill(0);for(let i=0;i<Z.length;i++){let z=[1,...indices.map(j=>Z[i][j])];for(let r=0;r<size;r++){b[r]+=W[i]*z[r]*Y[i];for(let c=0;c<size;c++)A[r][c]+=W[i]*z[r]*z[c];}}for(let j=1;j<size;j++)A[j][j]+=1;return solve(A,b);}
 const all=Array.from({length:d},(_,i)=>i),weights=fit(all),indices=all.sort((a,b)=>Math.abs(weights[b+1])-Math.abs(weights[a+1])).slice(0,3),coef=fit(indices);
 let ss=0,err=0;for(let i=0;i<Z.length;i++){let p=coef[0]+indices.reduce((s,j,k)=>s+coef[k+1]*Z[i][j],0);ss+=W[i]*(Y[i]-mean)**2;err+=W[i]*(Y[i]-p)**2;}
 return {source:'browser-adaptation',samples,r2:ss>0?1-err/ss:0,intercept:coef[0],localPrediction:coef.reduce((a,b)=>a+b,0),terms:indices.map((j,k)=>({feature:j,weight:coef[k+1],bin:xb[j]}))};
}
export function browserAnchors(model,x,shapValues,samples=256){
 const rng=seeded(43),target=+(predict(model,x)>=.5),d=x.length,bins=x.map((v,j)=>bin(v,model.quantiles[j]));
 const rank=Array.from({length:d},(_,i)=>i).sort((a,b)=>Math.abs(shapValues[b])-Math.abs(shapValues[a])).slice(0,8);
 const pools=rank.map(j=>model.train.filter(r=>bin(r[j],model.quantiles[j])===bins[j]).map(r=>r[j]));
 const candidates=rank.filter((j,k)=>pools[k].length>0);let best=null,beam=[[]];
 function evaluate(indices){let correct=0;for(let n=0;n<samples;n++){let row=[...model.train[Math.floor(rng()*model.train.length)]];for(let j of indices){let pool=pools[rank.indexOf(j)];row[j]=pool[Math.floor(rng()*pool.length)];}correct+=(+(predict(model,row)>=.5)===target);}
 let coverage=model.train.filter(r=>indices.every(j=>bin(r[j],model.quantiles[j])===bins[j])).length/model.train.length;
 return {indices,precision:correct/samples,coverage,target,source:'browser-anchor-style',samples,thresholdMet:correct/samples>=.95};}
 for(let size=1;size<=3;size++){let seen=new Set(),results=[];for(let old of beam)for(let j of candidates){if(old.includes(j))continue;let indices=[...old,j].sort((a,b)=>a-b),key=indices.join(',');if(seen.has(key))continue;seen.add(key);results.push(evaluate(indices));}
 let eligible=results.filter(r=>r.thresholdMet).sort((a,b)=>b.coverage-a.coverage);if(eligible.length)return {...eligible[0],bins};
 results.sort((a,b)=>b.precision-a.precision||b.coverage-a.coverage);if(results.length&&(!best||results[0].precision>best.precision))best=results[0];beam=results.slice(0,2).map(r=>r.indices);
 }
 return {...(best||{indices:[],precision:0,coverage:0,thresholdMet:false}),bins};
}
