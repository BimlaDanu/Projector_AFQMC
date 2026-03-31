import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.linalg import eigh

def mod(a, b):
    return (a + b) % b

def Bilayer_square_lattice(Lx, Ly, norb, nlayer, nl):
    loc = np.zeros((Lx, Ly, norb, nlayer), dtype = int)
    label = np.zeros((Lx*Ly*nl, 4), dtype = int)
    nbr =  np.zeros((Lx*Ly*nl, 2), dtype = int) 
    nnbr =  np.zeros((Lx*Ly*nl, 2), dtype = int)
    lnbr = np.zeros((Lx*Ly*nl, 2), dtype = int) 
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
            for layer in range(1):
                #print('layer = ',layer)
                s1 = loc[m][n][0][layer]
                nbr[s1][0] = loc[mod(m+1, Lx)][n][0][layer]
                nbr[s1][1] = loc[m][mod(n+1, Ly)][0][layer]
                
                nnbr[s1][0] = loc[mod(m+1, Lx)][mod(n+1, Lx)][0][layer]
                nnbr[s1][1] = loc[mod(m+1, Lx)][mod(n-1, Lx)][0][layer]
                
                lnbr[s1][0] = loc[m][n][0][layer ^ 1]                      
    return loc, label,  nbr, nnbr,  lnbr


def Bilayer_triangular_lattice(Lx, Ly, norb, nlayer, nl):
    loc = np.zeros((Lx, Ly, norb, nlayer), dtype = int)
    label = np.zeros((Lx*Ly*nl, 4), dtype = int)
    nbr =  np.zeros((Lx*Ly*nl, 3), dtype = int) 
    nnbr =  np.zeros((Lx*Ly*nl, 3), dtype = int)
    lnbr = np.zeros((Lx*Ly*nl, 3), dtype = int) 
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
                
                lnbr[s1][0] = loc[m][n][0][layer^1]                       
    return loc, label,  nbr, nnbr,  lnbr

def Bilayer_honeycomb_lattice(Lx, Ly, norb, nlayer):
    loc = np.zeros((Lx, Ly, norb,  nlayer), dtype = int)
    label = np.zeros((Lx * Ly * norb * nlayer, 4), dtype = int)
    nbr = np.zeros((Lx * Ly * norb * nlayer, 3), dtype = int)
    lnbr = np.zeros((Lx * Ly * norb * nlayer, 3), dtype = int)
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
                
                # Nearest-neighbour from sub-lattice A
                s1 = loc[m][n][0][layer]
                nbr[s1][0] = loc[m][n][1][layer]
                nbr[s1][1] = loc[mod(m-1, Lx)][n][1][layer]
                nbr[s1][2] = loc[m][mod(n-1, Ly)][1][layer]


                # Next- Nearest-neighbour from sub-lattice A
                nnbr[s1][0] = loc[mod(m+1, Lx)][n][0][layer]
                nnbr[s1][1] = loc[m][mod(n+1, Ly)][0][layer]
                nnbr[s1][2] = loc[mod(m-1, Lx)][mod(n+1, Ly)][0][layer]
                nnbr[s1][3] = loc[mod(m-1, Lx)][n][0][layer]
                nnbr[s1][4] = loc[m][mod(n-1, Ly)][0][layer]
                nnbr[s1][5] = loc[mod(m+1, Lx)][mod(n-1, Ly)][0][layer]
                
                lnbr[s1][0] = loc[m][n][0][layer ^ 1]
                lnbr[s1][1] = loc[mod(m-1, Lx)][n][0][layer ^ 1]
                lnbr[s1][2] = loc[m][mod(n-1, Ly)][0][layer ^ 1]


                # Nearest-neighbour from sub-lattice B
                s1 = loc[m][n][1][layer]
                nbr[s1][0] = loc[m][n][0][layer]
                nbr[s1][1] = loc[mod(m+1, Lx)][n][0][layer]
                nbr[s1][2] = loc[m][mod(n+1, Ly)][1][layer]

                lnbr[s1][0] = loc[m][n][1][layer ^ 1]
                lnbr[s1][1] = loc[mod(m+1, Lx)][n][1][layer ^ 1]
                lnbr[s1][2] = loc[m][mod(n+1, Ly)][1][layer ^ 1]

                # Next- Nearest-neighbour from sub-lattice B
                nnbr[s1][0] = loc[mod(m+1, Lx)][n][1][layer]
                nnbr[s1][1] = loc[m][mod(n+1, Ly)][1][layer]
                nnbr[s1][2] = loc[mod(m-1, Lx)][mod(n+1, Ly)][1][layer]
                nnbr[s1][3] = loc[mod(m-1, Lx)][n][1][layer]
                nnbr[s1][4] = loc[m][mod(n-1, Ly)][1][layer]
                nnbr[s1][5] = loc[mod(m+1, Lx)][mod(n-1, Ly)][1][layer]
    return  loc, label, nbr, lnbr, nnbr


def Bilayer_kagome_lattice(Lx, Ly, norb, nlayer):
    loc = np.zeros((Lx, Ly, norb,  nlayer), dtype = int)
    label = np.zeros((Lx * Ly * norb * nlayer, 4), dtype = int)
    nbr = np.zeros((Lx * Ly * norb * nlayer, 3), dtype = int)
    nnbr = np.zeros((Lx * Ly * norb * nlayer, 4), dtype = int)
    lnbr = np.zeros((Lx * Ly * norb * nlayer, 3), dtype = int)
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
                # from A sublattice
                s1 = loc[m][n][0][layer]
                nbr[s1][0] = loc[m][n][1][layer]
                nbr[s1][1] = loc[m][n][2][layer]
                nbr[s1][2] = loc[mod(m+1, Lx)][n][1][layer]
                lnbr[s1][0] = loc[m][n][0][layer^1]
                
                nnbr[s1][0] = loc[mod(m+1, Lx)][n][0][layer]
                nnbr[s1][1] = loc[mod(m-1, Lx)][n][0][layer]
                nnbr[s1][2] = loc[m][mod(n+1, Ly)][0][layer]
                nnbr[s1][3] = loc[m][mod(n-1, Ly)][0][layer]
                
                # from B sublattice
                s1 = loc[m][n][1][layer]
                nbr[s1][0] = loc[m][n][0][layer]
                nbr[s1][1] = loc[m][n][2][layer]
                nbr[s1][2] = loc[m][mod(n+1, Ly)][0][layer]
                lnbr[s1][0] = loc[m][n][1][layer^1]
                
                nnbr[s1][0] = loc[mod(m+1, Lx)][n][1][layer]
                nnbr[s1][1] = loc[mod(m-1, Lx)][n][1][layer]
                nnbr[s1][2] = loc[m][mod(n+1, Ly)][1][layer]
                nnbr[s1][3] = loc[m][mod(n-1, Ly)][1][layer]
                
    
                # from C sublattice
                s1 = loc[m][n][2][layer]
                nbr[s1][0] = loc[m][n][0][layer]
                nbr[s1][1] = loc[m][n][1][layer]
                nbr[s1][2] = loc[mod(m+1, Lx)][mod(n+1, Ly)][0][layer]
                lnbr[s1][0] = loc[m][n][2][layer^1]
                
                nnbr[s1][0] = loc[mod(m+1, Lx)][n][2][layer]
                nnbr[s1][1] = loc[mod(m-1, Lx)][n][2][layer]
                nnbr[s1][2] = loc[m][mod(n+1, Ly)][2][layer]
                nnbr[s1][3] = loc[m][mod(n-1, Ly)][2][layer]
    return  loc, label, nbr, lnbr, nnbr

def Anderson_hop_ham_bilayer_square(t, t_prime, V_ex, mu_f, mu_c, Lx, Ly, norb, nlayer): 
    ln = norb*nlayer
    loc, label,  nbr, nnbr,  lnbr =  Bilayer_square_lattice(Lx, Ly, norb, nlayer, ln)   
    ham = np.zeros((Lx*Ly*ln, Ly*Ly*ln), dtype = float)
    for m in range(Lx):
        for n in range(Ly):
            #hopping in layer1
            ls1 =  loc[m][n][0][0] # label [i][0], label[i][1]
            
            #chemical potential term
            ham[ls1][ls1] -= mu_c 
            
            # Nearest-neighbour hopping
            for ni in range(2):
                ls2 = nbr[ls1][ni]
                ham[ls1][ls2] -=   t
                ham[ls2][ls1] -=   t
                
            # Next- Nearest-neighbour hopping
            for ni in range(2):
                ls2 = nnbr[ls1][ni]
                ham[ls1][ls2] -=   t_prime
                ham[ls2][ls1] -=   t_prime
            
            #chemical potential term f
            lls2 =  loc[m][n][0][1] 
            ham[lls2][lls2] -=  mu_f
            
            #hopping in layer1 and layer2
            ls12 = lnbr[ls1][0]
            ham[ls1][ls12] +=   V_ex
            ham[ls12][ls1] +=   V_ex       
    return ham


def Anderson_projector_bilayer_square(t, V_ex, Lx, Ly, norb, nlayer): 
    epsl = 0.0001
    ln = norb*nlayer
    loc, label,  nbr, nnbr,  lnbr =  Bilayer_square_lattice(Lx, Ly, norb, nlayer, ln)   
    ham = np.zeros((Lx*Ly*ln, Ly*Ly*ln), dtype = float)
    for m in range(Lx):
        for n in range(Ly):
            #hopping in layer1
            ls1 =  loc[m][n][0][0] # label [i][0], label[i][1]
            s1x = nbr[ls1][0]
            s1y = nbr[ls1][1]
            #interaction in x directions
            ham[ls1][s1x] -=   t*(1.+ (-1)**(m+n)*epsl)
            ham[s1x][ls1] -=   t*(1.+ (-1)**(m+n)*epsl)
            #interaction in y directions
            ham[ls1][s1y] -=   t*(1.-epsl)
            ham[s1y][ls1] -=   t*(1.-epsl)
            

            #hopping in layer1 and layer2
            ls12 = lnbr[ls1][0]
            ham[ls1][ls12] +=   V_ex
            ham[ls12][ls1] +=   V_ex
    #print('matrix element of periodic Anderson model\n')
    #print_matrix_element(ham)
    return ham

def  Anderson_hop_ham_bilayer_triangular(t, t_prime,  V_ex, mu_c, mu_f, Lx, Ly, norb, nlayer): 
    ln = norb * nlayer
    loc, label,  nbr, nnbr,  lnbr =   Bilayer_triangular_lattice(Lx, Ly, norb, nlayer, ln)   
    ham = np.zeros((Lx*Ly*ln, Ly*Ly*ln), dtype = float)
    for m in range(Lx):
        for n in range(Ly):
            ls1 =  loc[m][n][0][0] # 
            
            # Nearest-neighbour hopping
            for ni in range(3):
                ls2 = nbr[ls1][ni]
                ham[ls1][ls2] -=   t
                ham[ls2][ls1] -=   t
                
            # Next-nearest-neighbour hopping
            for ni in range(3):
                ls2 = nnbr[ls1][ni]
                ham[ls1][ls2] -=   t_prime
                ham[ls2][ls1] -=   t_prime
                
            
            # Chemical potential c
            ham[ls1][ls1] -= mu_c
            
            # Chemical potential f
            lls2 =  loc[m][n][0][1] 
            ham[lls2][lls2] -=  mu_f
            
         #hopping in layer1 and layer2
            ls12 = lnbr[ls1][0]
            ham[ls1][ls12] +=   V_ex
            ham[ls12][ls1] +=   V_ex   
    return ham



def  Anderson_hop_ham_bilayer_triangular_square(t, t_prime, V_ex, mu_f, mu_c, Lx, Ly, norb, nlayer): 
    ln = norb*nlayer
    loc, label,  nbr, nnbr,  lnbr =  Bilayer_square_lattice(Lx, Ly, norb, nlayer, ln)   
    ham = np.zeros((Lx*Ly*ln, Ly*Ly*ln), dtype = float)
    for m in range(Lx):
        for n in range(Ly):
            #hopping in layer1
            ls1 =  loc[m][n][0][0] # label [i][0], label[i][1]
            s1x = nbr[ls1][0]
            s1y = nbr[ls1][1]
            
            #interaction in x directions
            ham[ls1][s1x] -=   t
            ham[s1x][ls1] -=   t
            
            #interaction in y directions
            ham[ls1][s1y] -=   t
            ham[s1y][ls1] -=   t
            
            ham[ls1][ls1] -= mu_c 
            
            #hopping in layer2
            ls2 =  loc[m][n][0][1] # label [i][0], label[i][1]
            s2x = nbr[ls2][0]
            s2y = nbr[ls2][1]
            
            #interaction in x directions #There is no hopping in second layers of localized moments
            ham[ls2][s2x] +=   0.
            ham[s2x][ls2] +=   0.
            
            #interaction in y directions
            ham[ls2][s2y] +=   0.
            ham[s2y][ls2] +=  0.
            ham[ls2][ls2] -=  mu_f
            
            #hopping in layer1 and layer2
            ls12 = lnbr[ls1][0]
            ham[ls1][ls12] +=   V_ex
            ham[ls12][ls1] +=   V_ex    
            
            s2x = nnbr[ls1, 0]
            s2y = nnbr[ls1, 1]
        
            # next-nearest neighbour hopping in x directions
            ham[ls1, s2x] -= t_prime*0
            ham[s2x, ls1] -= t_prime*0
            
            # next-nearest neighbour hopping in y directions
            ham[ls1, s2y] -= t_prime
            ham[s2y, ls1] -= t_prime
    return ham

def Anderson_projector_bilayer_triangular(t, V_ex, Lx, Ly, norb, nlayer): 
    epsl = 0.0001
    ln = norb * nlayer
    loc, label,  nbr, nnbr,  lnbr =  Bilayer_triangular_lattice(Lx, Ly, norb, nlayer, ln)   
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
            
            ls1 =  loc[m][n][0][0] # label [i][0], label[i][1]
            ls12 = lnbr[ls1][0]
            ham[ls1][ls12] +=   V_ex
            ham[ls12][ls1] +=   V_ex  
    return ham

def Anderson_hop_ham_bilayer_honeycomb(t, t_prime, mu_c, mu_f, V_ex, Lx, Ly, norb, nlayer):
    ham = np.zeros((Lx * Ly * norb*nlayer, Lx * Ly * norb*nlayer), dtype = float)
    loc, label,   nbr,   lnbr,    nnbr =  Bilayer_honeycomb_lattice(Lx, Ly, norb, nlayer)
    
    for n in range(Ly):
            for m in range(Lx):
                for layer in range(1):
                    ls1 = loc[m][n][0][layer]
                    
                    # chemical potential c
                    ham[ls1][ls1] -= mu_c 
                    
                                        
                    # Nearest-neighbour hopping
                    for ni in range(3):
                        ls2 = nbr[ls1][ni]
                        ham[ls1][ls2] -=   t
                        ham[ls2][ls1] -=   t
                        
                    # Next-Nearest-neighbour hopping
                    for ni in range(6):
                        ls2 = nnbr[ls1][ni]
                        ham[ls1][ls2] -=   t_prime
                        ham[ls2][ls1] -=   t_prime  
                                        
                    # Chemical potential
                    lls2 =  loc[m][n][0][1] # label [i][0], label[i][1]
                    ham[lls2][lls2] -=  mu_f
                    
                    #hopping in layer1 and layer2
                    ls12 = lnbr[ls1][0]
                    ham[ls1][ls12] +=   V_ex
                    ham[ls12][ls1] +=   V_ex             
    return ham



def Anderson_hop_ham_bilayer_kagome(t, t_prime, mu_c, mu_f, V_ex, Lx, Ly, norb, nlayer):
    ham = np.zeros((Lx * Ly * norb*nlayer, Lx * Ly * norb*nlayer), dtype = float)
    loc, label,   nbr,   lnbr,    nnbr =  Bilayer_kagome_lattice(Lx, Ly, norb, nlayer)
    for n in range(Ly):
            for m in range(Lx):
                for layer in range(1):
                    ls1 = loc[m][n][0][layer]
                    # Chemical potential c
                    ham[ls1][ls1] -= mu_c 
                    
                    # Nearest-neighbour hopping
                    for ni in range(3):
                        ls2 = nbr[ls1][ni]
                        ham[ls1][ls2] -=   t
                        ham[ls2][ls1] -=   t
                        
                    # Next-Nearest-neighbour hopping
                    for ni in range(4):
                        ls2 = nnbr[ls1][ni]
                        ham[ls1][ls2] -=   t_prime
                        ham[ls2][ls1] -=   t_prime 
           
                    # Chemical potential f
                    lls2 =  loc[m][n][0][1]
                    ham[lls2][lls2] -=  mu_f
                    
                    # Hopping in layer1 and layer2
                    ls12 = lnbr[ls1][0]
                    ham[ls1][ls12] +=   V_ex
                    ham[ls12][ls1] +=   V_ex             
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


def call_Hamiltonian_Bilayer(lattice_type, t, t_prime, V_ex, mu_f, mu_c, Lx, Ly, norb, nlayer, Per):
    if lattice_type == 'Bilayer_Square':
        ham =  Anderson_hop_ham_bilayer_square(t, t_prime, V_ex, mu_f, mu_c, Lx, Ly, norb, nlayer)
    elif lattice_type == 'Bilayer_Triangular':
        ham =  Anderson_hop_ham_bilayer_triangular(t, t_prime, V_ex, mu_f, mu_c, Lx, Ly, norb, nlayer)
        #ham =  Anderson_hop_ham_bilayer_triangular_square(t, t_prime, V_ex, mu_f, mu_c,  Lx, Ly, norb, nlayer)
    elif lattice_type == 'Bilayer_Honeycomb':
        ham =  Anderson_hop_ham_bilayer_honeycomb(t, t_prime, V_ex, mu_f, mu_c, Lx, Ly, norb, nlayer)
    elif lattice_type == 'Bilayer_Kagome':
        ham =  Anderson_hop_ham_bilayer_kagome(t, t_prime, V_ex, mu_f, mu_c, Lx, Ly, norb, nlayer)
    else:
        raise ValueError("Invalid lattice type. Supported types: 'Square', 'Triangular', 'Honeycomb', 'Kagome', '. ") 
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
#    t =1.
#    t_prime = 0.
#    V_ex = 0.5
#    mu_f = 0.
#    mu_c = 0.
#    Lx = 3
#    Ly = Lx
#    nlayer = 2
#    Per = True
#    #lattice_type = 'Bilayer_Square'
#    lattice_type = 'Bilayer_Triangular'
    #lattice_type = 'Bilayer_Honeycomb'
    #lattice_type = 'Bilayer_Kagome'

