import json, sys, collections
sb = json.load(open(sys.argv[1]))
texts = [ (s.get("name","")+". "+(s.get("mechanism","") or "")) for s in sb ]
names = [s.get("name","") for s in sb]
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    V = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2).fit_transform(texts)
    S = cosine_similarity(V)
    n=len(sb)
    for thr in (0.60,0.70,0.80):
        # union-find connected components over edges >= thr (i<j)
        parent=list(range(n))
        def f(x):
            while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
            return x
        import numpy as np
        iu=np.triu_indices(n,1)
        edges=0
        for i,j in zip(*iu):
            if S[i,j]>=thr:
                ri,rj=f(i),f(j)
                if ri!=rj: parent[ri]=rj
                edges+=1
        comp=collections.defaultdict(list)
        for i in range(n): comp[f(i)].append(i)
        clusters=[c for c in comp.values() if len(c)>1]
        in_dup=sum(len(c) for c in clusters)
        print(f"thr={thr}: {len(clusters)} near-dup clusters, {in_dup} skills ({100*in_dup/n:.0f}%) in a cluster, largest={max((len(c) for c in clusters),default=0)}, edges={edges}")
    # show the clusters at 0.70 with example names
    thr=0.70
    parent=list(range(n))
    def f2(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    iu=np.triu_indices(n,1)
    for i,j in zip(*iu):
        if S[i,j]>=thr:
            ri,rj=f2(i),f2(j)
            if ri!=rj: parent[ri]=rj
    comp=collections.defaultdict(list)
    for i in range(n): comp[f2(i)].append(i)
    clusters=sorted([c for c in comp.values() if len(c)>1], key=lambda c:-len(c))
    print(f"\n--- clusters at thr=0.70 (top 8) ---")
    for c in clusters[:8]:
        print(f"  [{len(c)}] " + " | ".join(names[k][:32] for k in c[:5]) + (" ..." if len(c)>5 else ""))
except Exception as e:
    print("sklearn path failed:", repr(e))
