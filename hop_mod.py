def Hop_mod_symm(self, GR, Bk_root, inv_Bk_root,l):
    N_FL = self.N_FL
    for nf in range(N_FL):
        GR[:,:,nf] = Bk_root[l, :,:,nf] @ GR[:,:,nf] @ inv_Bk_root[l,:,:,nf]
    return GR