from mpi4py import MPI
import numpy as np
import traceback

def simulate_magnetization_dynamics(hper, hpara):
        # Constants
    gam = 1.76e11  # 1/(s T)
    ele = 1.602e-19  # C
    mu_0 = 1.25663706143592e-06  # N/A^2
    hbar = 1.055e-34  # J.s
    alp = 0.01
    Hani = 1000 / (4 * np.pi) * 1e3
    Ms = 1500 * 1e3
    tfm = 1.6e-9
    radius = 60e-9
    S = np.pi * radius ** 2
    runtime = 5e-10
    tstep = 2e-12
    J0 = 2 * ele * mu_0 * Ms * tfm * Hani / hbar
    cpara = 0.05
    totstep = round(runtime / tstep)
    spin_raw = np.array([0, -1, -1])
    zaxis = np.array([0, 0, 1])
    spin = spin_raw / np.linalg.norm(spin_raw)
    Hext = np.array([0, 0, -1000])
    hext = Hext / Hani / (4 * np.pi) * 1e3
    ts1 = tstep * gam * mu_0 * Hani  # time step

    mmx = np.zeros(totstep)
    mmy = np.zeros(totstep)
    mmz = np.zeros(totstep)
    m_init = np.array([0, 0, 1])

    mmx[0], mmy[0], mmz[0] = m_init

    for ct1 in range(totstep - 1):
        mm1 = np.array([mmx[ct1], mmy[ct1], mmz[ct1]])
        mmm = mm1
        costheta = np.dot(mmm, zaxis) / (np.linalg.norm(mmm) * np.linalg.norm(zaxis))
        hh = costheta * np.array([0, 0, 1]) + hext
        olddmdt = -np.cross(mmm, hh) - hpara * np.cross(mmm, np.cross(mmm, spin)) - hper * np.cross(mmm, spin)
        dmdt = -np.cross(mmm, hh) + alp * np.cross(mmm, olddmdt) - hpara * np.cross(mmm, np.cross(mmm, spin)) - hper * np.cross(mmm, spin)
        kk1 = dmdt

        for _ in range(3):  # A simplified Runge-Kutta method
            mmm = mm1 + kk1 * ts1 / 2
            costheta = np.dot(mmm, zaxis) / (np.linalg.norm(mmm) * np.linalg.norm(zaxis))
            hh = costheta * np.array([0, 0, 1]) + hext
            olddmdt = -np.cross(mmm, hh) - hpara * np.cross(mmm, np.cross(mmm, spin)) - hper * np.cross(mmm, spin)
            dmdt = -np.cross(mmm, hh) + alp * np.cross(mmm, olddmdt) - hpara * np.cross(mmm, np.cross(mmm, spin)) - hper * np.cross(mmm, spin)
            kk1 = dmdt 

        mn1 = mm1 + ts1 / 6 * (kk1 + 2 * kk1 + 2 * kk1 + kk1) 
        mn1 /= np.linalg.norm(mn1)
        mmx[ct1 + 1], mmy[ct1 + 1], mmz[ct1 + 1] = mn1

    return [mmx, mmy, mmz]

def generate_parameter_space(start, stop, step):
    return np.arange(start, stop + step, step)

def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    hper_space = generate_parameter_space(0, 1, 0.1)
    hpara_space = generate_parameter_space(0, 1, 0.1)
    total_simulations = len(hper_space) * len(hpara_space)
    simulations_per_process = total_simulations // size
    start_index = rank * simulations_per_process
    end_index = start_index + simulations_per_process
    if rank == size - 1:
        end_index += total_simulations % size
    results = []

    for index in range(start_index, end_index):
        hper_index = index // len(hpara_space)
        hpara_index = index % len(hpara_space)
        hper = hper_space[hper_index]
        hpara = hpara_space[hpara_index]
        result = simulate_magnetization_dynamics(hper, hpara)
        results.append((hper, hpara, result[2][-1]))
        
    # Gather results from all processes
    all_results = comm.gather(results, root=0)
    if rank == 0:
        for process_results in all_results:
            for hper, hpara, result in process_results:
                print(f'hper: {hper:.4f}, hpara: {hpara:.4f}, result: {result:.4f}')

if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(traceback.format_exc())

