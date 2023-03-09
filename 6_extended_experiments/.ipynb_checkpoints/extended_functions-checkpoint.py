from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdmolfiles
import pymol
from pymol import cmd

# torsion angle determination in a mol file
def enumerateTorsions(mol):
    # this function comes from https://sourceforge.net/p/rdkit/mailman/message/34554615/
    torsionSmarts = '[!$(*#*)&!D1]~[!$(*#*)&!D1]'
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = mol.GetSubstructMatches(torsionQuery)
    torsionList = []
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = mol.GetBondBetweenAtoms(idx2, idx3)
        jAtom = mol.GetAtomWithIdx(idx2)
        kAtom = mol.GetAtomWithIdx(idx3)
        if (((jAtom.GetHybridization() != Chem.HybridizationType.SP2)
           and (jAtom.GetHybridization() != Chem.HybridizationType.SP3))
           or ((kAtom.GetHybridization() != Chem.HybridizationType.SP2)
           and (kAtom.GetHybridization() != Chem.HybridizationType.SP3))):
            continue
        for b1 in jAtom.GetBonds():
            if (b1.GetIdx() == bond.GetIdx()):
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if ((b2.GetIdx() == bond.GetIdx())
                    or (b2.GetIdx() == b1.GetIdx())):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                 # skip 3-membered rings
                if (idx4 == idx1):
                    continue
                torsionList.append((idx1, idx2, idx3, idx4))
    return torsionList

# calculate bond in a mol
def get_bond_length(mol):
    l_bonds = []
    for i in mol.GetBonds():
        l_bonds.append((i.GetBondType().__str__(), 
              mol.GetAtomWithIdx(i.GetBeginAtomIdx()).GetSymbol(), i.GetBeginAtomIdx(), 
              mol.GetAtomWithIdx(i.GetEndAtomIdx()).GetSymbol(), i.GetEndAtomIdx()))
    dist_mtx = Chem.Get3DDistanceMatrix(mol)
    len_bond = [dist_mtx[x[2], x[-1]] for x in l_bonds]
    bonds = [list(x) +[y] for x,y in zip(l_bonds, len_bond)]
    return bonds

# calculate dihedral angle in a mol
def get_dihedral_angle(mol):
    l_dihedral = enumerateTorsions(mol)
    l_out = []
    for (i,j,k,l) in l_dihedral:
        r = rdMolTransforms.GetDihedralDeg(mol.GetConformer(0),i,j,k,l)
        l_out.append(r)
    return l_out

# align post processed ligand to reference cocrystal ligand
def align(LIG, PDB, PDB_ref, pocket):
    cmd.load(PDB_ref, f'pro_ref')
    cmd.remove('(hydro)')
    cmd.select('ref', f'pro_ref and resi {pocket} and backbone')
    
    cmd.load(PDB, 'pro')
    cmd.load(LIG, 'lig')
    cmd.create('comp','pro or lig')
    cmd.remove('(hydro)')
    cmd.select('complex', f'comp and resi {pocket} and backbone')
    cmd.align('complex', 'ref')
    cmd.save(LIG.replace('.mol2', '_0.mol2'), 'lig')
    cmd.delete("all")
    
def align_gro(GRO, PDB_ref, pocket):
    cmd.load(PDB_ref, f'pro_ref')
    cmd.remove('(hydro)')
    cmd.select('ref', f'pro_ref and resi {pocket} and backbone')
    
    cmd.load(GRO, 'comp')
    cmd.remove('(hydro)')
    cmd.select('complex', f'comp and resi {pocket} and backbone')
    cmd.align('complex', 'ref')
    cmd.select('LIG', 'comp and resn LIG')
    cmd.select('PRO', 'comp and polymer')
    
    cmd.save(GRO.replace('.gro', '.mol2'), 'LIG')
    cmd.delete("all")