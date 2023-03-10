from __future__ import print_function, division

import torch

import torch.nn as nn
from torch.nn import ParameterList
from torch.nn.parameter import Parameter

class PotentialModel(nn.Module):
    def __init__(self, lammps_data, movable_idx, special_rotation = None, macro_mode = None):
        super(PotentialModel, self).__init__()

        self.atom_type = torch.LongTensor( lammps_data.atom_type )
        self.atom_charge = Parameter(torch.Tensor( lammps_data.atom_charge ), requires_grad = False)
        self.atom_mass = Parameter(torch.Tensor( lammps_data.atom_mass ), requires_grad=False)

        self.bond_idx = torch.LongTensor( lammps_data.bond_idx )
        self.angle_idx = torch.LongTensor( lammps_data.angle_idx )
        self.dihedral_idx = torch.LongTensor( lammps_data.dihedral_idx )
        self.improper_idx = torch.LongTensor( lammps_data.improper_idx )

        self.epsilon = Parameter(torch.Tensor( lammps_data.epsilon ), requires_grad=False)
        self.sigma = Parameter(torch.Tensor( lammps_data.sigma ), requires_grad=False)
        
        self.bond_harmonic_idx = torch.LongTensor( lammps_data.bond_harmonic_idx )
        self.bond_harmonic_coeffs = Parameter(torch.Tensor( lammps_data.bond_harmonic_coeffs ), requires_grad=False)
        self.angle_harmonic_idx = torch.LongTensor( lammps_data.angle_harmonic_idx )
        self.angle_harmonic_coeffs = Parameter(torch.Tensor( lammps_data.angle_harmonic_coeffs ), requires_grad=False)
        self.angle_charmm_idx = torch.LongTensor( lammps_data.angle_charmm_idx )
        self.angle_charmm_coeffs = Parameter(torch.Tensor( lammps_data.angle_charmm_coeffs ), requires_grad=False)
        self.dihedral_multiharm_idx = torch.LongTensor( lammps_data.dihedral_multiharm_idx )
        self.dihedral_multiharm_coeffs = Parameter(torch.Tensor( lammps_data.dihedral_multiharm_coeffs ), requires_grad=False)
        self.dihedral_charmm_idx = torch.LongTensor( lammps_data.dihedral_charmm_idx )
        self.dihedral_charmm_coeffs = Parameter(torch.Tensor( lammps_data.dihedral_charmm_coeffs ), requires_grad=False)
        self.improper_harmonic_idx = torch.LongTensor( lammps_data.improper_harmonic_idx )
        self.improper_harmonic_coeffs = Parameter(torch.Tensor( lammps_data.improper_harmonic_coeffs ), requires_grad=False)

        na = self.atom_type.shape[0]
        self.sb_mask = Parameter(torch.ones(na,na), requires_grad=False)
        # Gromacs-LAMMPS files have special_bonds set to:
        # 1st neighbor = 0, 2nd neighbor = 0.0, 3rd neighbor = 1.0
        # The rest of LJ & Coulomb interactions are calculated normally
        self.sb_mask[self.bond_idx[:,0], self.bond_idx[:,1]] = 0
        self.sb_mask[self.bond_idx[:,1], self.bond_idx[:,0]] = 0
        self.sb_mask[self.angle_idx[:,0], self.angle_idx[:,2]] = 0
        self.sb_mask[self.angle_idx[:,2], self.angle_idx[:,0]] = 0
        self.sb_mask[self.dihedral_idx[:,0], self.dihedral_idx[:,3]] = 1.0
        self.sb_mask[self.dihedral_idx[:,3], self.dihedral_idx[:,0]] = 1.0
        

        self.ij_mask = Parameter(torch.nonzero(torch.triu(torch.ones(na,na, dtype=int), diagonal=1), as_tuple=False), requires_grad=False)
        #self.coulomb_coeff = 9e9 * 1.602e-19 * 1.602e-19 / 1e-10 / 4.184 / 1e3 * 6.02e23
        self.coulomb_coeff = 332.33

        xyz = torch.Tensor( lammps_data.atom_pos )
        self.device = xyz.device
        self.attach_init_inputs(xyz, movable_idx, special_rotation=special_rotation, macro_mode=macro_mode)

        self.bonded_only = False
        return

    def to(self, device):
        super(PotentialModel, self).to(device)
        self.device = device
        return self

    def attach_init_inputs(self, xyz, movable_idx_list, special_rotation = None, macro_mode = None):
        # Ensure movable_idx content is unique
        # Ideally movable_idx is ordered, but it is fine if it is not ordered. 
        movable_dict = dict()
        for movable_idx in movable_idx_list:
            for idx in movable_idx:
                if idx in movable_dict: raise Exception('Movable atom index',idx,'appears more than once')
                movable_dict[idx] = True
        na = xyz.shape[0]
        fixed_idx = []
        for i in range(na):
            if not (i in movable_dict):
                fixed_idx.append(i)
        self.movable_idx_list = ParameterList( [Parameter(torch.LongTensor(movable_idx), requires_grad=False)
                                                for movable_idx in movable_idx_list] )
        self.fixed_idx = torch.LongTensor(fixed_idx)
        self.movable_pos_list = ParameterList( [Parameter(xyz[movable_idx,:], requires_grad = True)
                                                for movable_idx in self.movable_idx_list] )
        self.fixed_pos = Parameter(xyz[self.fixed_idx,:], requires_grad = False)

        self.translation_list = Parameter(torch.zeros(len(movable_idx_list),1,3), requires_grad = True)
        self.rotation_list = Parameter(torch.zeros(len(movable_idx_list),3), requires_grad = True)

        # If special rotation centers are defined
        # special_rotation will be dictionary of movable_idx_list group -> bonded atom idx
        if special_rotation != None:
            assert len(special_rotation)<=len(movable_idx_list)
            for group_id in special_rotation:
                assert group_id in range(len(movable_idx_list))
                assert special_rotation[group_id][0] in range(self.atom_type.shape[0])
                assert special_rotation[group_id][1] in range(self.atom_type.shape[0])
                assert special_rotation[group_id][2] in range(2)
            self.special_rotation_idx = Parameter(torch.LongTensor([(i,j[0],j[1],j[2]) for i,j in special_rotation.items()]), requires_grad=False)
            self.special_rotation_list = Parameter(torch.zeros(len(special_rotation),1), requires_grad = True)
        else:
            self.special_rotation_idx = None
            self.special_rotation_list = None
            
        if macro_mode != None:
            flat_group = [item for sublist in macro_mode for item in sublist]
            unique_group = list(set(flat_group))
            assert len(flat_group) == len(unique_group)
            for group_id in flat_group:
                assert group_id in range(len(movable_idx_list))
            self.macro_mode_idx = ParameterList( [Parameter(torch.LongTensor(group_list), requires_grad=False)
                                                  for group_list in macro_mode] )
            self.macro_mode_translation_list = Parameter(torch.zeros(len(macro_mode),1,3), requires_grad = True)
            self.macro_mode_rotation_list = Parameter(torch.zeros(len(macro_mode),3), requires_grad = True)
        else:
            self.macro_mode_idx = None
            self.macro_mode_translation_list = None
            self.macro_mode_rotation_list = None                            

        self.to(self.device)
        self.atom_pos = self.arrange_atom_pos(self.movable_pos_list, self.fixed_pos)
        return

    def arrange_atom_pos(self, movable_pos_list, fixed_pos):
        if self.special_rotation_idx != None:
            movable_pos_list = self.axis_rotate(movable_pos_list, fixed_pos)

        movable_pos_list = self.micro_rotate_translate(movable_pos_list)

        if self.macro_mode_idx != None:
            movable_pos_list = self.macro_rotate_translate(movable_pos_list)
        
        na = sum([movable_pos.shape[0] for movable_pos in movable_pos_list]) + fixed_pos.shape[0]
        atom_pos = torch.zeros(na,3).to(self.device)
        for i in range(len(movable_pos_list)):
            atom_pos[self.movable_idx_list[i],:] = movable_pos_list[i]
        atom_pos[self.fixed_idx,:] = fixed_pos

        return atom_pos

    def micro_rotate_translate(self, in_xyz_list):
        zero = torch.LongTensor([0]).to(self.device)
        nm = len(in_xyz_list)
        ng = torch.LongTensor([len(i) for i in in_xyz_list]).to(self.device)
        na = torch.sum(ng)
        in_xyz = torch.cat([in_xyz_list[i] for i in range(nm)], dim=0)
        com_xyz = torch.cat([in_xyz_list[i].mean(dim=0).expand(ng[i],3) for i in range(nm)], dim=0)
        trans_xyz = torch.cat([self.translation_list[i,:,:].expand(ng[i],3) for i in range(nm)], dim=0)
        rot_angles = torch.repeat_interleave(self.rotation_list, ng, dim=0)
        a, b, c = rot_angles[:,0], rot_angles[:,1], rot_angles[:,2]
        sin_a, cos_a = torch.sin(a), torch.cos(a)
        sin_b, cos_b = torch.sin(b), torch.cos(b)
        sin_c, cos_c = torch.sin(c), torch.cos(c)
        Ra = torch.zeros(na,3,3).to(self.device)
        Rb = torch.zeros(na,3,3).to(self.device)
        Rc = torch.zeros(na,3,3).to(self.device)
        Ra[:,0,0] = cos_a
        Ra[:,0,1] = -sin_a
        Ra[:,1,0] = sin_a
        Ra[:,1,1] = cos_a
        Ra[:,2,2] = 1
        Rb[:,0,0] = cos_b
        Rb[:,0,2] = sin_b
        Rb[:,2,0] = -sin_b
        Rb[:,2,2] = cos_b
        Rb[:,1,1] = 1
        Rc[:,1,1] = cos_c
        Rc[:,1,2] = -sin_c
        Rc[:,2,1] = sin_c
        Rc[:,2,2] = cos_c
        Rc[:,0,0] = 1
        R = torch.matmul(Ra, torch.matmul(Rb, Rc))
        frame_xyz = in_xyz - com_xyz
        rot_xyz = torch.matmul(frame_xyz.unsqueeze(1), R.transpose(1,2)).view(na,3)
        out_xyz = rot_xyz + com_xyz + trans_xyz
        
        indices = torch.cumsum(torch.cat([zero, ng], dim=0), dim=0)
        out_xyz_list = [out_xyz[ indices[i]:indices[i+1], :] for i in range(nm)]

        return out_xyz_list

    def macro_rotate_translate(self, in_xyz_list):
        zero = torch.LongTensor([0]).to(self.device)
        nm = len(in_xyz_list)
        ng = torch.LongTensor([len(i) for i in in_xyz_list]).to(self.device)
        na = torch.sum(ng)
        nmac = len(self.macro_mode_idx)
        indices = torch.cumsum(torch.cat([zero, ng], dim=0), dim=0)
        in_xyz = torch.cat([in_xyz_list[i] for i in range(nm)], dim=0)
        com_xyz = torch.cat([in_xyz_list[i].mean(dim=0).expand(ng[i],3) for i in range(nm)], dim=0)
        trans_xyz = torch.zeros(na,3).to(self.device)
        rot_angles = torch.zeros(na,3).to(self.device)
        for i in range(nmac):
            macro_movable_idx = torch.cat([ torch.arange(indices[j],indices[j+1]) for j in self.macro_mode_idx[i] ])
            trans_xyz[macro_movable_idx,:] = self.macro_mode_translation_list[i]
            rot_angles[macro_movable_idx,:] = self.macro_mode_rotation_list[i]
            # Now we need to replace com_xyz of the macro groups with the macro centers, just for the ligand
            if i == (nmac-1):
                com_xyz[macro_movable_idx,:] = in_xyz[macro_movable_idx,:].mean(dim=0)

        a, b, c = rot_angles[:,0], rot_angles[:,1], rot_angles[:,2]
        sin_a, cos_a = torch.sin(a), torch.cos(a)
        sin_b, cos_b = torch.sin(b), torch.cos(b)
        sin_c, cos_c = torch.sin(c), torch.cos(c)
        Ra = torch.zeros(na,3,3).to(self.device)
        Rb = torch.zeros(na,3,3).to(self.device)
        Rc = torch.zeros(na,3,3).to(self.device)
        Ra[:,0,0] = cos_a
        Ra[:,0,1] = -sin_a
        Ra[:,1,0] = sin_a
        Ra[:,1,1] = cos_a
        Ra[:,2,2] = 1
        Rb[:,0,0] = cos_b
        Rb[:,0,2] = sin_b
        Rb[:,2,0] = -sin_b
        Rb[:,2,2] = cos_b
        Rb[:,1,1] = 1
        Rc[:,1,1] = cos_c
        Rc[:,1,2] = -sin_c
        Rc[:,2,1] = sin_c
        Rc[:,2,2] = cos_c
        Rc[:,0,0] = 1
        R = torch.matmul(Ra, torch.matmul(Rb, Rc))
        frame_xyz = in_xyz - com_xyz
        rot_xyz = torch.matmul(frame_xyz.unsqueeze(1), R.transpose(1,2)).view(na,3)
        out_xyz = rot_xyz + com_xyz + trans_xyz

        out_xyz_list = [out_xyz[ indices[i]:indices[i+1], :] for i in range(nm)]

        return out_xyz_list

    def translate(self, in_xyz_list):
        return [in_xyz_list[i] + self.translation_list[i] for i in range(len(in_xyz_list))]

    def rotate(self, in_xyz_list):
        zero = torch.LongTensor([0]).to(self.device)
        nm = len(in_xyz_list)
        ng = torch.LongTensor([len(i) for i in in_xyz_list]).to(self.device)
        na = torch.sum(ng)
        in_xyz = torch.cat([in_xyz_list[i] for i in range(nm)], dim=0)
        com_xyz = torch.cat([in_xyz_list[i].mean(dim=0).expand(ng[i],3) for i in range(nm)], dim=0)
        rot_angles = torch.repeat_interleave(self.rotation_list, ng, dim=0)
        a, b, c = rot_angles[:,0], rot_angles[:,1], rot_angles[:,2]
        sin_a, cos_a = torch.sin(a), torch.cos(a)
        sin_b, cos_b = torch.sin(b), torch.cos(b)
        sin_c, cos_c = torch.sin(c), torch.cos(c)
        Ra = torch.zeros(na,3,3).to(self.device)
        Rb = torch.zeros(na,3,3).to(self.device)
        Rc = torch.zeros(na,3,3).to(self.device)
        Ra[:,0,0] = cos_a
        Ra[:,0,1] = -sin_a
        Ra[:,1,0] = sin_a
        Ra[:,1,1] = cos_a
        Ra[:,2,2] = 1
        Rb[:,0,0] = cos_b
        Rb[:,0,2] = sin_b
        Rb[:,2,0] = -sin_b
        Rb[:,2,2] = cos_b
        Rb[:,1,1] = 1
        Rc[:,1,1] = cos_c
        Rc[:,1,2] = -sin_c
        Rc[:,2,1] = sin_c
        Rc[:,2,2] = cos_c
        Rc[:,0,0] = 1
        R = torch.matmul(Ra, torch.matmul(Rb, Rc))
        frame_xyz = in_xyz - com_xyz
        rot_xyz = torch.matmul(frame_xyz.unsqueeze(1), R.transpose(1,2)).view(na,3)
        out_xyz = rot_xyz + com_xyz
        
        indices = torch.cumsum(torch.cat([zero, ng], dim=0), dim=0)
        out_xyz_list = [out_xyz[ indices[i]:indices[i+1], :] for i in range(nm)]

        return out_xyz_list

    def anchor_rotate(self, in_xyz):
        ns = self.special_rotation_idx.shape[0]
        # gi = group_id, ai = anchor_idx
        in_xyz_list = [(in_xyz[self.movable_idx_list[gi],:], in_xyz[ai,:]) for (gi, ai) in self.special_rotation_idx]
        com_xyz_list = [in_xyz_list[i][0] - in_xyz_list[i][1] for i in range(ns)]
        a,b,c = self.special_rotation_list[:,0], self.special_rotation_list[:,1], self.special_rotation_list[:,2]
        Ra = torch.Tensor([[0,0,0],[0,0,0],[0,0,1]]).repeat(ns,1,1).to(self.device)
        Rb = torch.Tensor([[0,0,0],[0,1,0],[0,0,0]]).repeat(ns,1,1).to(self.device)
        Rc = torch.Tensor([[1,0,0],[0,0,0],[0,0,0]]).repeat(ns,1,1).to(self.device)
        Ra[:,0,0] = torch.cos(a)
        Ra[:,0,1] = -torch.sin(a)
        Ra[:,1,0] = torch.sin(a)
        Ra[:,1,1] = torch.cos(a)
        Rb[:,0,0] = torch.cos(b)
        Rb[:,0,2] = torch.sin(b)
        Rb[:,2,0] = -torch.sin(b)
        Rb[:,2,2] = torch.cos(b)
        Rc[:,1,1] = torch.cos(c)
        Rc[:,1,2] = -torch.sin(c)
        Rc[:,2,1] = torch.sin(c)
        Rc[:,2,2] = torch.cos(c)
        R_list = [torch.matmul( Ra[i], torch.matmul(Rb[i], Rc[i]) ) for i in range(ns)]
        rot_xyz_list = [torch.matmul(com_xyz_list[i], R_list[i].transpose(0,1)) for i in range(ns)]
        out_xyz_list = [rot_xyz_list[i] + in_xyz_list[i][1] for i in range(ns)]
        out_xyz = in_xyz.clone()
        for i in range(ns):
            out_xyz[ self.movable_idx_list[self.special_rotation_idx[i,0]], : ] = out_xyz_list[i]
        return out_xyz        

    def axis_rotate(self, movable_pos_list, fixed_pos):
        na = sum([movable_pos.shape[0] for movable_pos in movable_pos_list]) + fixed_pos.shape[0]
        atom_pos = torch.zeros(na,3).to(self.device)
        for i in range(len(movable_pos_list)):
            atom_pos[self.movable_idx_list[i],:] = movable_pos_list[i]
        atom_pos[self.fixed_idx,:] = fixed_pos
        zero = torch.LongTensor([0]).to(self.device)

        rot_xyz_list = [movable_pos for movable_pos in movable_pos_list]

        ns = torch.LongTensor([self.movable_idx_list[gi].shape[0] for gi in self.special_rotation_idx[:,0]]).to(self.device)
        C = torch.cat([atom_pos[self.movable_idx_list[gi],:] for gi in self.special_rotation_idx[:,0]], dim=0)
        A = torch.repeat_interleave(atom_pos[self.special_rotation_idx[:,1],:], ns, dim=0)
        B = torch.repeat_interleave(atom_pos[self.special_rotation_idx[:,2],:], ns, dim=0)
        theta = torch.repeat_interleave(self.special_rotation_list[:,[0]], ns, dim=0).expand(torch.sum(ns),3)
        U = B-A
        R = C-A
        u = U/torch.linalg.norm(U,axis=1).view(-1,1)
        Z = (R*u).sum(axis=1).view(-1,1)*u
        x = R-Z
        y = torch.cross(u, x, dim=1)
        rot_pos = A + Z + x*torch.cos(theta) + y*torch.sin(theta)
        indices = torch.cumsum(torch.cat([zero, ns], dim=0), dim=0)
        for i in range(len(ns)):
            gi = self.special_rotation_idx[i,0]
            rot_xyz_list[gi] = rot_pos[ indices[i]:indices[i+1], :]

        return rot_xyz_list

    def jolt_movable_atoms(self, seed = None, max_translation = 5.0, max_rotation = 3.14, ignore_last = False):
        if not (seed==None):
            torch.manual_seed(seed)
        self.translation_list.requires_grad, self.rotation_list.requires_grad = False, False
        translation_shape = [i for i in self.translation_list.shape] #self.translation_list.shape
        rotation_shape = [i for i in self.rotation_list.shape] #self.rotation_list.shape
        if ignore_last:
            translation_shape[0] += -1 #translation_shape = [translation_shape[0] - 1, translation_shape[1], translation_shape[2]]
            rotation_shape[0] += -1 #rotation_shape = [rotation_shape[0] - 1, rotation_shape[1]]
            self.translation_list[:-1,:,:] = (torch.rand(translation_shape) - 0.5) * max_translation * 2
            self.rotation_list[:-1,:] = (torch.rand(rotation_shape) - 0.5) * max_rotation * 2
        else:
            self.translation_list[:,:,:] = torch.rand(translation_shape) * max_translation
            self.rotation_list[:,:] = torch.rand(rotation_shape) * max_rotation
        self.translation_list.requires_grad, self.rotation_list.requires_grad = True, True

        # Take care of special rotation
        if self.special_rotation_idx != None:
            rotation_shape = [i for i in self.special_rotation_list.shape] #self.special_rotation_list.shape
            self.special_rotation_list.requires_grad = False
            self.special_rotation_list[:,:] = (torch.rand(rotation_shape) - 0.5) * max_rotation * 2
            for j in torch.where(self.special_rotation_idx[:,3]==1)[0]:
                if torch.rand(1) > 0.5 and max_rotation != 0.0:
                    self.special_rotation_list[j,:] += 3.1416   # with 50% probability, initially rotate these groups 180 degrees
            self.special_rotation_list.requires_grad = True

        # Take care of macro mode. We only reset this to 0. We will not give a kick as the groups are too big.
        if self.macro_mode_idx != None:
            self.macro_mode_translation_list.requires_grad = False
            self.macro_mode_rotation_list.requires_grad = False
            self.macro_mode_translation_list[:,:,:] = 0
            self.macro_mode_rotation_list[:,:] = 0
            self.macro_mode_translation_list.requires_grad = True
            self.macro_mode_rotation_list.requires_grad = True            

        return            

    def random_full_rotation(self, rotated_indices, seed = None):
        if not (seed==None):
            torch.manual_seed(seed)
        all_pos = torch.Tensor(self.atom_pos.detach().cpu().numpy())
        rot_pos = all_pos[ rotated_indices,: ]
        rot_com = rot_pos.mean(dim=0)
        pi = torch.acos(torch.Tensor([-1]))
        theta, phi, z = 2*pi*torch.rand([1]), 2*pi*torch.rand([1]), 2*torch.rand([1])
        sin_t, sin_p = torch.sin(theta), torch.sin(phi)
        cos_t, cos_p = torch.cos(theta), torch.cos(phi)
        r = torch.sqrt(z)
        V = torch.Tensor([sin_p*r, cos_p*r, torch.sqrt(2-z)])
        S = torch.Tensor([V[0]*cos_t - V[1]*sin_t, V[0]*sin_t + V[1]*cos_t])
        R = torch.Tensor([ [ V[0]*S[0]-cos_t, V[0]*S[1]-sin_t, V[0]*V[2] ],
                                     [ V[1]*S[0]+sin_t, V[1]*S[1]-cos_t, V[1]*V[2] ],
                                     [ V[2]*S[0], V[2]*S[1], 1-z ] ])
        rot_pos = rot_com + torch.matmul(rot_pos-rot_com, R.transpose(0,1))
        all_pos[ rotated_indices,: ] = rot_pos
        self.attach_init_inputs(all_pos, [rotated_indices] )
        return        
    
    def forward(self):
        #atom_pos needs to be updated because movable_pos is updated on each epoch
        self.atom_pos = self.arrange_atom_pos(self.movable_pos_list, self.fixed_pos)
        na = self.atom_pos.shape[0]
        all_dist = torch.linalg.norm(self.atom_pos.unsqueeze(0).expand(na,na,3) - \
                                     self.atom_pos.unsqueeze(1).expand(na,na,3),
                                     dim=2)
        self.E_bond = self.calculate_E_bond(all_dist)
        self.E_angle = self.calculate_E_angle(self.atom_pos, all_dist)
        self.E_dihedral = self.calculate_E_dihedral(self.atom_pos)
        self.E_improper = self.calculate_E_improper(self.atom_pos)
        self.E_LJ = self.calculate_E_LJ(all_dist)
        self.E_coulomb = self.calculate_E_coulomb(all_dist)
        
##        print('Energy bond\t: ',self.E_bond)
##        print('Energy angle\t: ',self.E_angle)
##        print('Energy dihedral\t: ',self.E_dihedral)
##        print('Energy improper\t: ',self.E_improper)
##        print('Energy LJ\t: ',self.E_LJ)
##        print('Energy Coulomb\t: ',self.E_coulomb)
        
        # right now this energy is kcal/mol
        if self.bonded_only:
            E_total = self.E_bond + self.E_angle + self.E_dihedral + self.E_improper + 0.2 * (self.E_LJ + self.E_coulomb)
        else:
            E_total = self.E_bond + self.E_angle + self.E_dihedral + self.E_improper + self.E_LJ + self.E_coulomb
##        print('Energy total\t: ',E_total)
        return E_total

    def calculate_E_bond(self, all_dist):
        d_bond = all_dist[ self.bond_idx[:,0],self.bond_idx[:,1] ]

        # Define output
        E_bond = 0
        
        # Harmonic
        idx = self.bond_harmonic_idx
        coeffs = self.bond_harmonic_coeffs
        if not(coeffs.shape[0] == 0):
            E_bond_harmonic = coeffs[:,0] * ((d_bond[idx] - coeffs[:,1])**2)
            E_bond += E_bond_harmonic.sum()
        
        return E_bond

    def _d2r(self, angle_deg):
        torch_pi = torch.acos(torch.zeros(1))*2
        angle_rad = angle_deg / 180.0 * torch_pi.to(angle_deg.device)
        return angle_rad

    def calculate_E_angle(self, atom_pos, all_dist):
        v1 = atom_pos[ self.angle_idx[:,0] ] - atom_pos[ self.angle_idx[:,1] ]
        v2 = atom_pos[ self.angle_idx[:,2] ] - atom_pos[ self.angle_idx[:,1] ]
        temp1 = torch.sum(v1*v2, dim=1)
        temp2 = torch.linalg.norm(v1, dim=1) * torch.linalg.norm(v2, dim=1)
        d_cos = temp1 / temp2
        d_cos = torch.clamp(d_cos, min=-0.999999, max=0.999999)
        angle = torch.acos( d_cos )
        dist = all_dist[ self.angle_idx[:,0], self.angle_idx[:,2] ]

        # Define output
        E_angle = 0
        
        # Harmonic
        idx = self.angle_harmonic_idx
        coeffs = self.angle_harmonic_coeffs
        if not (coeffs.shape[0] == 0):
            ref_angle = self._d2r(coeffs[:,1])
            E_angle_harmonic = coeffs[:,0] * ((angle[idx] - ref_angle)**2)
            E_angle += E_angle_harmonic.sum()

        # Charmm
        idx = self.angle_charmm_idx
        coeffs = self.angle_charmm_coeffs
        if not (coeffs.shape[0] == 0):
            ref_angle = self._d2r(coeffs[:,1])
            E_angle_charmm = coeffs[:,0] * ((angle[idx] - ref_angle)**2) +\
                             coeffs[:,2] * ((dist[idx] - coeffs[:,3])**2)
            E_angle += E_angle_charmm.sum()        

        return E_angle

    def calculate_E_dihedral(self, atom_pos):
        v12 = atom_pos[ self.dihedral_idx[:,0] ] - atom_pos[ self.dihedral_idx[:,1] ]
        v32 = atom_pos[ self.dihedral_idx[:,2] ] - atom_pos[ self.dihedral_idx[:,1] ]
        v43 = atom_pos[ self.dihedral_idx[:,3] ] - atom_pos[ self.dihedral_idx[:,2] ]
        v123 = torch.cross(v12,v32, dim=1)
        v234 = torch.cross(-v32,v43, dim=1)
        temp1 = torch.sum(v123*v234, dim=1)
        temp2 = torch.linalg.norm(v123, dim=1) * torch.linalg.norm(v234, dim=1)
        d_cos = temp1 / temp2
        d_cos = torch.clamp(d_cos, min=-0.999999, max=0.999999)

        # Define output
        E_dihedral = 0
        
        # Multi/harmonic
        idx = self.dihedral_multiharm_idx
        coeffs = self.dihedral_multiharm_coeffs
        if not(coeffs.shape[0] == 0):
            temp_d_cos = d_cos[idx]
            E_dihedral_multiharm = coeffs[:,0] +\
                                   coeffs[:,1] * temp_d_cos +\
                                   coeffs[:,2] * torch.pow(temp_d_cos,2) +\
                                   coeffs[:,3] * torch.pow(temp_d_cos,3) +\
                                   coeffs[:,4] * torch.pow(temp_d_cos,4)
            E_dihedral += E_dihedral_multiharm.sum()
        
        # Charmm
        idx = self.dihedral_charmm_idx
        coeffs = self.dihedral_charmm_coeffs
        if not(coeffs.shape[0] == 0):
            d_acos = torch.acos(d_cos[idx])
            ref_angle = self._d2r(coeffs[:,2])
            E_dihedral_charmm = coeffs[:,0] * (1 + torch.cos(coeffs[:,1]*d_acos - ref_angle)) * coeffs[:,3]

            # There is nan grad problem associated with weight of 0 in CHARMM force field coeffs[:,3]
            # Somehow Gromacs-LAMMPS conversion produces these 0-contribution force field.
            # Ignoring the contribution eliminates this problem
            if (coeffs[:,3]**2).sum() == 0:
                E_dihedral = E_dihedral
            else:
                E_dihedral += E_dihedral_charmm.sum()

        return E_dihedral

    def calculate_E_improper(self, atom_pos):
        if (self.improper_idx.shape[0] == 0):
            return 0
        v12 = atom_pos[ self.improper_idx[:,0] ] - atom_pos[ self.improper_idx[:,1] ]
        v32 = atom_pos[ self.improper_idx[:,2] ] - atom_pos[ self.improper_idx[:,1] ]
        v43 = atom_pos[ self.improper_idx[:,3] ] - atom_pos[ self.improper_idx[:,2] ]
        v123 = torch.cross(v12,v32, dim=1)
        v234 = torch.cross(-v32,v43, dim=1)
        temp1 = torch.sum(v123*v234, dim=1)
        temp2 = torch.linalg.norm(v123, dim=1) * torch.linalg.norm(v234, dim=1)
        d_cos = temp1 / temp2
        d_cos = torch.clamp(d_cos, min=-0.999999, max=0.999999)
        
        # Define output
        E_improper = 0
        
        # Harmonic
        idx = self.improper_harmonic_idx
        coeffs = self.improper_harmonic_coeffs
        if not(coeffs.shape[0] == 0):
            d_acos = torch.acos(d_cos[idx])
            ref_angle = self._d2r(coeffs[:,1])
            E_improper_harmonic = coeffs[:,0] * ((d_acos - ref_angle)**2)
            E_improper += E_improper_harmonic.sum()

        return E_improper

    def calculate_E_LJ(self, all_dist):
        # these indices get all pairs with 0 < rij < 9.0, but both (i,j) and (j,i) are included
        indices = torch.nonzero( (all_dist-4.5)**2 < 4.5**2, as_tuple=False)
        # with this, only 1 of each reciprocal pairs are included
        indices = indices[torch.nonzero( indices[:,0]<indices[:,1], as_tuple=False)[:,0],:]
        type_i = self.atom_type[ indices[:,0] ]
        type_j = self.atom_type[ indices[:,1] ]
        eps = self.epsilon[ type_i, type_j ]
        sigma = self.sigma[ type_i, type_j ]
        r = all_dist[ indices[:,0],indices[:,1] ]
        frac = (sigma/r)**6
        mask = self.sb_mask[ indices[:,0],indices[:,1] ]
        E_LJ = 4*eps*( frac**2 - frac ) * mask
        return E_LJ.sum()

    def calculate_E_coulomb(self, all_dist):
        r = all_dist[ self.ij_mask[:,0],self.ij_mask[:,1] ]
        charge_i = self.atom_charge[ self.ij_mask[:,0] ]
        charge_j = self.atom_charge[ self.ij_mask[:,1] ]
        mask = self.sb_mask[ self.ij_mask[:,0],self.ij_mask[:,1] ]
        E_coulomb = self.coulomb_coeff * charge_i * charge_j * mask / r
        return E_coulomb.sum()
    
