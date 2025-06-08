from Simulations.data_simulation_JM import simulate_JM_base
n_sim = 1
I = 1000
obstime = [0,1,2,3,4,5,6,7,8,9,10]
landmark_times = [1,2,3,4,5]
pred_windows = [1,2,3]
scenario = "none" # ["none", "interaction", "nonph"]

from sklearn.preprocessing import MinMaxScaler

data_all = simulate_JM_base(I=I, obstime=obstime, opt=scenario, seed=n_sim)
data = data_all[data_all.obstime <= data_all.time]

data.to_pickle('data/simulated_data.pkl')