import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.linalg import eigh

def mod(a, b):
    return (a + b) % b

def one_dimensional_chain(length, per):
    #if per == True:
    loc = np.zeros((length), dtype = int)
    nbr = np.zeros((length), dtype = int)
    nnbr = np.zeros((length), dtype = int)
    #elif per == False:
    #    loc = np.zeros((length-1), dtype = int)
    #    nbr = np.zeros((length-1), dtype = int)
    #    nnbr = np.zeros((length-1), dtype = int)
    
    
    site = 0
    
    for m in range(length):
        if per==True:
            loc[site] = mod(m+1, length) 
        elif per==False and m <= length-1: 
            loc[site] = m
        site += 1
        
    for m in range(length):
        s1 = loc[m]
        if per==True:
            nbr[s1] = loc[mod(m+1, Lx)]
        elif per==False and m < length-1: 
            nbr[s1] = loc[m+1]
            
            
        if per==True:
            nnbr[s1] = loc[mod(m+2, Lx)]
        elif per==False and m < length-2:
            nnbr[s1] = loc[m+2]     
    #print('loc,  nbr, nnbr =', loc,  nbr, nnbr)
    return loc,  nbr, nnbr


def Square_lattice(Lx, Ly, norb, nlayer, nl):
    loc = np.zeros((Lx, Ly, norb, nlayer), dtype = int)
    label = np.zeros((Lx*Ly*nl, 4), dtype = int)
    nbr =  np.zeros((Lx*Ly*nl, 2), dtype = int) 
    nnbr =  np.zeros((Lx*Ly*nl, 2), dtype = int)
    site = 0
    
    for m in range (Lx):
        for n in range (Ly):
            for ab in range(norb):
                for layer in range(nlayer):
                    loc[m][n][ab][layer] = site
                    label [site][0] = m
                    label [site][1] = n
                    label [site][2] = ab
                    label [site][3] = layer
                    site +=1  
                    
                                                                      
    for m in range(Lx):
        for n in range(Ly):   
            for layer in range(nlayer):
                #print('layer = ',layer)
                s1 = loc[m][n][0][layer]
                nbr[s1][0] = loc[mod(m+1, Lx)][n][0][layer]
                nbr[s1][1] = loc[m][mod(n+1, Ly)][0][layer]
                
                nnbr[s1][0] = loc[mod(m+1, Lx)][mod(n+1, Lx)][0][layer]
                nnbr[s1][1] = loc[mod(m+1, Lx)][mod(n-1, Lx)][0][layer]                            
    return loc, label, nbr, nnbr


def Triangular_lattice(Lx, Ly, norb, nlayer, nl):
    loc = np.zeros((Lx, Ly, norb, nlayer), dtype = int)
    label = np.zeros((Lx*Ly*nl, 4), dtype = int)
    nbr =  np.zeros((Lx*Ly*nl, 3), dtype = int) 
    nnbr =  np.zeros((Lx*Ly*nl, 3), dtype = int)
    site = 0
    for m in range (Lx):
        for n in range (Ly):
            for ab in range(norb):
                for layer in range(nlayer):
                    loc[m][n][ab][layer] = site
                    label [site][0] = m
                    label [site][1] = n
                    label [site][2] = ab
                    label [site][3] = layer
                    site +=1                                                                
    for m in range(Lx):
        for n in range(Ly):   
            for layer in range(nlayer):
                #print('layer = ',layer)
                s1 = loc[m][n][0][layer]
                nbr[s1][0] = loc[m][mod(n+1, Lx)][0][layer]
                nbr[s1][1] = loc[mod(m+1, Lx)][n][0][layer]
                nbr[s1][2] = loc[mod(m+1, Lx)][mod(n-1, Ly)][0][layer]
                
                nnbr[s1][0] = loc[m][mod(n+2, Lx)][0][layer]
                nnbr[s1][2] = loc[mod(m+2, Lx)][n][0][layer]
                nnbr[s1][2] = loc[mod(m+2, Lx)][mod(n-2, Ly)][0][layer]                           
    return loc, label,  nbr, nnbr

def Honeycomb_lattice(Lx, Ly, norb, nlayer):
    loc = np.zeros((Lx, Ly, norb,  nlayer), dtype = int)
    label = np.zeros((Lx * Ly * norb * nlayer, 4), dtype = int)
    nbr = np.zeros((Lx * Ly * norb * nlayer, 3), dtype = int)
    nnbr = np.zeros((Lx * Ly * norb * nlayer, 6), dtype = int)
    site = 0
    for n in range(Ly):
        for m in range(Lx):
            for ab in range(norb):
                for layer in range(nlayer):
                    loc[m][n][ab][layer] = site
                    label[site][0] = m 
                    label[site][1] = n  
                    label[site][2] = ab
                    label[site][3] = layer
                    site += 1
    for n in range(Ly):
        for m in range(Lx):
            for layer in range(nlayer):
                # Nearest neighbour from A sublattice
                s1 = loc[m][n][0][layer]
                nbr[s1][0] = loc[m][n][1][layer]
                nbr[s1][1] = loc[mod(m-1, Lx)][n][1][layer]
                nbr[s1][2] = loc[m][mod(n-1, Ly)][1][layer]
                
                #Next-Nearest neighbour from A sublattice
                nnbr[s1][0] = loc[mod(m+1, Lx)][n][0][layer]
                nnbr[s1][1] = loc[m][mod(n+1, Ly)][0][layer]
                nnbr[s1][2] = loc[mod(m-1, Lx)][mod(n+1, Ly)][0][layer]
                nnbr[s1][3] = loc[mod(m-1, Lx)][n][0][layer]
                nnbr[s1][4] = loc[m][mod(n-1, Ly)][0][layer]
                nnbr[s1][5] = loc[mod(m+1, Lx)][mod(n-1, Ly)][0][layer]

                # Nearest neighbour from B sublattice
                s1 = loc[m][n][1][layer]
                nbr[s1][0] = loc[m][n][0][layer]
                nbr[s1][1] = loc[mod(m+1, Lx)][n][0][layer]
                nbr[s1][2] = loc[m][mod(n+1, Ly)][1][layer]
                
                #Next-Nearest neighbour from B sublattice
                nnbr[s1][0] = loc[mod(m+1, Lx)][n][1][layer]
                nnbr[s1][1] = loc[m][mod(n+1, Ly)][1][layer]
                nnbr[s1][2] = loc[mod(m-1, Lx)][mod(n+1, Ly)][1][layer]
                nnbr[s1][3] = loc[mod(m-1, Lx)][n][1][layer]
                nnbr[s1][4] = loc[m][mod(n-1, Ly)][1][layer]
                nnbr[s1][5] = loc[mod(m+1, Lx)][mod(n-1, Ly)][1][layer]
    return  loc, label, nbr, nnbr


def Kagome_lattice(Lx, Ly, norb, nlayer):
    loc = np.zeros((Lx, Ly, norb,  nlayer), dtype = int)
    label = np.zeros((Lx * Ly * norb * nlayer, 4), dtype = int)
    nbr = np.zeros((Lx * Ly * norb * nlayer, 3), dtype = int)
    nnbr = np.zeros((Lx * Ly * norb * nlayer, 4), dtype = int)
    site = 0
    for n in range(Ly):
        for m in range(Lx):
            for ab in range(norb):
                for layer in range(nlayer):
                    loc[m][n][ab][layer] = site
                    label[site][0] = m 
                    label[site][1] = n  
                    label[site][2] = ab
                    label[site][3] = layer
                    site += 1
    for n in range(Ly):
        for m in range(Lx):
            for layer in range(nlayer):
                
                #from A sublattice
                s1 = loc[m][n][0][layer]
                nbr[s1][0] = loc[m][n][1][layer]
                nbr[s1][1] = loc[m][n][2][layer]
                nbr[s1][2] = loc[mod(m+1, Lx)][n][1][layer]
                
                nnbr[s1][0] = loc[mod(m+1, Lx)][n][0][layer]
                nnbr[s1][1] = loc[mod(m-1, Lx)][n][0][layer]
                nnbr[s1][2] = loc[m][mod(n+1, Ly)][0][layer]
                nnbr[s1][3] = loc[m][mod(n-1, Ly)][0][layer]
                
                
                
                #from B sublattice
                s1 = loc[m][n][1][layer]
                nbr[s1][0] = loc[m][n][0][layer]
                nbr[s1][1] = loc[m][n][2][layer]
                nbr[s1][2] = loc[m][mod(n+1, Ly)][0][layer]
                
                nnbr[s1][0] = loc[mod(m+1, Lx)][n][1][layer]
                nnbr[s1][1] = loc[mod(m-1, Lx)][n][1][layer]
                nnbr[s1][2] = loc[m][mod(n+1, Ly)][1][layer]
                nnbr[s1][3] = loc[m][mod(n-1, Ly)][1][layer]
                
                
                
                #from C sublattice
                s1 = loc[m][n][2][layer]
                nbr[s1][0] = loc[m][n][0][layer]
                nbr[s1][1] = loc[m][n][1][layer]
                nbr[s1][2] = loc[mod(m+1, Lx)][mod(n+1, Ly)][0][layer]
                
                nnbr[s1][0] = loc[mod(m+1, Lx)][n][2][layer]
                nnbr[s1][1] = loc[mod(m-1, Lx)][n][2][layer]
                nnbr[s1][2] = loc[m][mod(n+1, Ly)][2][layer]
                nnbr[s1][3] = loc[m][mod(n-1, Ly)][2][layer]
    return  loc, label, nbr, nnbr



def Hubbard_hop_ham_1d_chain(t, t_prime, mu, Lx, Per): 
    loc, nbr, nnbr = one_dimensional_chain(Lx,   Per) 
    ham = np.zeros((Lx, Lx), dtype = float)
    for m in range(Lx):
        s1 =  loc[m]
        s1x = nbr[s1] 
        
        # First neighbour interaction
        ham[s1][s1x] -=   t
        ham[s1x][s1] -=   t
        
        s2x = nnbr[s1] 
        
        #Second neighbour interaction
        ham[s1][s2x] -=   t_prime
        ham[s2x][s1] -=   t_prime
    return ham

def Hubbard_hop_ham_square(t, t_prime, mu, Lx, Ly, norb, nlayer): 
    ln = norb*nlayer
    loc, label,  nbr, nnbr =  Square_lattice(Lx, Ly, norb, nlayer, ln)   
    ham = np.zeros((Lx*Ly*ln, Ly*Ly*ln), dtype = float)
    for m in range(Lx):
        for n in range(Ly):
            s1 =  loc[m][n][0][0] # label [i][0], label[i][1]
    
            # Nearest-neighbour hopping
            for ni in range(2):
                s2 = nbr[s1][ni]
                ham[s1][s2] -=   t
                ham[s2][s1] -=   t
                
            # Next-nearest-neighbour hopping
            for ni in range(2):
                s2 = nnbr[s1][ni]
                ham[s1][s2] -=   t_prime
                ham[s2][s1] -=   t_prime
            
            # Chemical potential term
            ham[s1][s1] -= mu
    return ham

def Hubbard_hop_ham_triangular_square(t, t_prime, mu, Lx, Ly, norb, nlayer): 
    ln = norb*nlayer
    loc, label,  nbr, nnbr =  Square_lattice(Lx, Ly, norb, nlayer, ln)   
    ham = np.zeros((Lx*Ly*ln, Ly*Ly*ln), dtype = float)
    for m in range(Lx):
        for n in range(Ly):
            s1 =  loc[m][n][0][0] # label [i][0], label[i][1]
            s1x = nbr[s1][0]
            s1y = nbr[s1][1]
            
            #interaction in x directions
            ham[s1][s1x] -=   t
            ham[s1x][s1] -=   t
            
            #interaction in y directions
            ham[s1][s1y] -=   t
            ham[s1y][s1] -=   t
            
            ham[s1][s1] -= mu
            
            s2x = nnbr[s1, 0]
            s2y = nnbr[s1, 1]
        
            # next-nearest neighbour hopping in x directions
            ham[s1, s2x] -= t_prime*0
            ham[s2x, s1] -= t_prime *0
            
            # next-nearest neighbour hopping in y directions
            ham[s1, s2y] -= t_prime  
            ham[s2y, s1] -= t_prime 
    return ham


def Hubbard_projector_square(t, Lx, Ly, norb, layer_num): 
    epsl = 0.0001
    ln = norb*layer_num
    loc, label,  nbr, nnbr =  Square_lattice(Lx, Ly, norb, layer_num, ln)   
    ham = np.zeros((Lx*Ly*ln, Ly*Ly*ln), dtype = float)
    for m in range(Lx):
        for n in range(Ly):
            #hopping in layer1
            s1 =  loc[m][n][0][0] # label [i][0], label[i][1]
            s1x = nbr[s1][0]
            s1y = nbr[s1][1]
            #interaction in x directions
            ham[s1][s1x] -=   t*(1.+ (-1)**(m+n)*epsl)
            ham[s1x][s1] -=   t*(1.+ (-1)**(m+n)*epsl)
            #interaction in y directions
            ham[s1][s1y] -=   t*(1.-epsl)
            ham[s1y][s1] -=   t*(1.-epsl)

    return ham


def Hubbard_hop_ham_triangular(t, t_prime, mu, Lx, Ly, norb, nlayer): 
    ln = norb*nlayer
    loc, label,  nbr, nnbr =  Triangular_lattice(Lx, Ly, norb, nlayer, ln)   
    ham = np.zeros((Lx*Ly*ln, Ly*Ly*ln), dtype = float)
    for m in range(Lx):
        for n in range(Ly):
            s1 =  loc[m][n][0][0] # label [i][0], label[i][1]
            
            # Nearest-neighbour hopping
            for ni in range(3):
                s2 = nbr[s1][ni]
                ham[s1][s2] -=   t
                ham[s2][s1] -=   t
            
            #chemical potential term
            ham[s1][s1] -= mu
            
            # Next-Nearest-neighbour hopping
            for ni in range(3):
                s2 = nnbr[s1][ni]
                ham[s1][s2] -=   t_prime
                ham[s2][s1] -=   t_prime
    return ham


def Hubbard_projector_triangular(t, Lx, Ly, norb, layer_num): 
    epsl = 0.0001
    ln = norb*layer_num
    loc, label,  nbr, nnbr =  Triangular_lattice(Lx, Ly, norb, layer_num, ln)   
    ham = np.zeros((Lx*Ly*ln, Ly*Ly*ln), dtype = float)
    for m in range(Lx):
        for n in range(Ly):
            #hopping in layer1
            s1 =  loc[m][n][0][0] # label [i][0], label[i][1]
            
            s11 = nbr[s1][0]
            s12 = nbr[s1][1]
            s13 = nbr[s1][2]

            ham[s1][s11] -=   t*(1.+ (-1)**(m+n)*epsl)
            ham[s11][s1] -=   t*(1.+ (-1)**(m+n)*epsl)
    
            ham[s1][s12] -=   t*(1.-epsl)
            ham[s12][s1] -=   t*(1.-epsl)
            
            ham[s1][s13] -=   t*(1.-epsl)
            ham[s13][s1] -=   t*(1.-epsl)
    return ham


def Hubbard_hop_ham_honeycomb(t, t_prime, mu, Lx, Ly, norb, nlayer):
    ham = np.zeros((Lx * Ly * norb*nlayer, Lx * Ly * norb*nlayer), dtype = float)
    loc, label,   nbr,  nnbr =  Honeycomb_lattice(Lx, Ly, norb, nlayer)
    for n in range(Ly):
            for m in range(Lx):
                for layer in range(nlayer):
                    s1 = loc[m][n][0][layer]
                    
                    #chemical potential
                    ham[s1][s1] -= mu 
                    
                    # Nearest-neighbour hopping
                    for ni in range(3):
                        s2 = nbr[s1][ni]
                        ham[s1][s2] -=   t
                        ham[s2][s1] -=   t
                        
                    # Next-Nearest-neighbour hopping
                    for ni in range(6):
                        s2 = nnbr[s1][ni]
                        ham[s1][s2] -=   t_prime
                        ham[s2][s1] -=   t_prime                          
    return ham



def Hubbard_hop_ham_kagome(t, t_prime, mu, Lx, Ly, norb, nlayer):
    ham = np.zeros((Lx * Ly * norb*nlayer, Lx * Ly * norb*nlayer), dtype = float)
    loc, label,   nbr,  nnbr =  Kagome_lattice(Lx, Ly, norb, nlayer)
    for n in range(Ly):
            for m in range(Lx):
                for layer in range(nlayer):
                    s1 = loc[m][n][0][layer]
                    
                    #chemical potential
                    ham[s1][s1] -= mu 
                    
                    # Nearest-neighbour hopping
                    for ni in range(3):
                        s2 = nbr[s1][ni]
                        ham[s1][s2] -=   t
                        ham[s2][s1] -=   t
                        
                    # Next-Nearest-neighbour hopping
                    for ni in range(4):
                        s2 = nnbr[s1][ni]
                        ham[s1][s2] -=   t_prime
                        ham[s2][s1] -=   t_prime            
    return ham

def call_Hamiltonian(lattice_type, t, t_prime, mu, Lx, Ly, norb, nlayer, Per):
    if lattice_type == 'Square':
        ham =  Hubbard_hop_ham_square(t, t_prime, mu, Lx, Ly, norb, nlayer)
    elif lattice_type == 'Triangular':
        ham = Hubbard_hop_ham_triangular(t, t_prime, mu, Lx, Ly, norb, nlayer)
        #ham =  Hubbard_hop_ham_triangular_square(t, t_prime, mu, Lx, Ly, norb, nlayer)
    elif lattice_type == 'Honeycomb':
        ham = Hubbard_hop_ham_honeycomb(t, t_prime, mu, Lx, Ly, norb, nlayer)
    elif lattice_type == 'Kagome':
        ham = Hubbard_hop_ham_kagome(t, t_prime, mu, Lx, Ly, norb, nlayer)
    elif lattice_type == '1d_chain':
        ham  =Hubbard_hop_ham_1d_chain(t, t_prime, mu, Lx, Per)
    else:
        raise ValueError("Invalid lattice type. Supported types: 'Square', 'Triangular', 'Honeycomb', 'Kagome', '1d_chain'. ") 
    return ham

def print_hamiltonian(ham):
    rows, cols = ham.shape
    for i in range(rows):
        for j in range(cols):
            print(f"{ham[i, j]:2.1f}", end=" ")
        print()  # Newline after each row

def plot_eigenvalues(eigenvalues):
    plt.figure(figsize=(8, 6))
    plt.plot(eigenvalues, 'o', linestyle='--', color='b', markersize=5)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues vs Index')
    plt.grid(True)
    plt.show()
    

#if __name__ == "__main__":
    #t =1.
    #t_prime = 0.
    #mu = 0.
    #Lx = 3
    #Ly = Lx
    #nlayer = 1
    #Per = True
    #Per = False
   # lattice_type = 'Square'
    #lattice_type = 'Triangular'
    #lattice_type = 'Honeycomb'
    #lattice_type = 'Kagome'
    #lattice_type = '1d_chain'

    #if  lattice_type == 'Square' or   lattice_type == 'Triangular' or   lattice_type == '1d_chain':
    #    norb = 1
    #elif lattice_type == 'Honeycomb':
    #     norb = 2
    #elif lattice_type == 'Kagome':
    #     norb = 3
          
    #ham = call_Hamiltonian(lattice_type, t, t_prime, mu, Lx, Ly, norb, nlayer, Per)
    #print('ham.shape =', ham.shape)
    #print_hamiltonian(ham)
    #eigenvalues, _ = np.linalg.eigh(ham)
    #print(eigenvalues)
    #plot_eigenvalues(eigenvalues)