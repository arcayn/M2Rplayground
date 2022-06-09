# this file will randomly generate polytopes and output them in plucker coordinate
# form

import random
import itertools


def plucker(P, vl=None):
    #M1 = Matrix([[1,1],[0,1]])
    if vl is None: vl = list(P.vertices())
    M_ = Matrix([b for b in vl]).transpose()
    M = M_.right_kernel().matrix()
    rows,_ = M.dimensions()
    minors = M.minors(rows)
    return ProjectiveSpace(len(minors) - 1, ZZ)(minors)

def gen_all_plucker(dim):
    assert dim in (2, 3)
    for idx in range(15 if dim == 2 else 4318):
        pt = ReflexivePolytope(dim, idx)
        if pt.lattice().submodule(pt.vertices()).index_in(pt.lattice()) != 1:
            continue
        for vl in itertools.permutations(list(pt.vertices())):
            yield (plucker(pt, vl), idx)
    return

def gen_whole_plucker_sample(dim):
    assert dim in (2, 3)
    r = []
    for idx in range(15 if dim == 2 else 4318):
        pt = ReflexivePolytope(dim, idx)
        if pt.lattice().submodule(pt.vertices()).index_in(pt.lattice()) != 1:
            continue
            
        rr = set()
        for vl in itertools.permutations(list(pt.vertices())):
            rr.add(plucker(pt, vl))
        r.append((rr,idx))
    return r

def gen_random_plucker_sample(dim):
    assert dim in (2, 3)

    r = []
    for idx in range(15 if dim == 2 else 4318):
        print (idx)
        pt = ReflexivePolytope(dim, idx)
        if pt.lattice().submodule(pt.vertices()).index_in(pt.lattice()) != 1:
            continue
        if len(pt.vertices()) < 5:
            continue
        
        rr = set()
        if len(pt.vertices()) <= 5:
            for vl in itertools.permutations(list(pt.vertices())):
                rr.add(plucker(pt, vl))
        else:
            print ("H")
            t = list(pt.vertices())
            random.shuffle(t)
            cnt = 0
            while len(rr) < 75:
                rr.add(plucker(pt, t))
                random.shuffle(t)
                cnt += 1
                if cnt > 1000:
                    print ("BAD", cnt)
                    break
        
        if len(rr) < 75:
            continue
        rl = random.sample(list(rr), 75)
        
        r.append((rl, idx))
    return r

def map_lattice_pt(pt, M):
    V = [M * v for v in pt.vertices()]
    return LatticePolytope(V)

def gen_reflexive_polytope(dim):
    # will work on generating arbitrary reflexive polytopes
    # but for now we focus on generating isomorphisms to the PALP db
    assert dim in (2, 3)

    idx = random.randint(0, 15 if dim == 2 else 4318)
    base_pt = ReflexivePolytope(dim, idx)
    """
    transform = [random.randint(0, 100) for _ in range(dim + 1)]

    pt = base_pt
    for e,M in zip(transform, GL(dim, ZZ).as_matrix_group().gens()):
        pt = map_lattice_pt(pt, (M ^ e)) 
    """

    M = random_matrix(ZZ,dim,dim)
    while abs(M.det()) != 1:
        M = random_matrix(ZZ,dim,dim)
    pt = map_lattice_pt(base_pt, M)
    
    # make sure we have actually performed a CoB
    assert pt.index() == idx
    return pt, idx

def gen_primitive_polytope(dim, n):
    assert n >= dim + 1


def write_plucker_sample(dim):
    sample = gen_whole_plucker_sample(dim)
    sample = [[[list(s) for s in S],i] for S,i in sample]
    print (sample)
    with open("plucker_s.txt", "wb") as f:
        f.write(str(sample).encode())

def plotPoly(P):
    rad = 3
    plot = [[' ' for _ in range(2 * rad + 1)] for _ in range(2 * rad + 1)]
    for x,y in P.vertices():
        plot[rad - y][rad + x] = '.'
    plot[rad][rad] = 'O'
    for pp in plot:
        print (''.join(pp))

plotPoly(ReflexivePolytope(2, 11))
plotPoly(ReflexivePolytope(2, 8))
for i in range(16):
    print (i, plucker(ReflexivePolytope(2, i)))