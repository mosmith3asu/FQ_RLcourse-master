#!/usr/bin/env python3

#from RL_Controller import MemoryHandler
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime
import copy
def get_MJT(tf,xf,t0=0,x0=0,N=100):
	"""
	𝜏 is the normalized time and equal to 𝑡/𝑡
	"""
	#t = np.linspace(t0,tf,N)
	dt = (tf-t0)/N
	tau = np.linspace(0,1,N)
	xt = x0 + (xf - x0)*(6*np.power(tau,5) - 15*np.power(tau,4) +10* np.power(tau,3))
	
	vt = np.gradient(xt)/dt
	at = np.gradient(vt)/dt
	Jt = np.gradient(at)/dt
	return xt,vt,at,Jt

class MemoryHandler(object):
	def __init__(self):
		self.sticky_names = ['Episode Reward','Episode Length','Episode Energy']
		self.nonsticky_names = ['Damping','Velocity','Force'] # 'Jerk'
		self.names = self.sticky_names +self.nonsticky_names
		
		now = datetime.now()
		self.date_label = now.strftime("-Date_%m_%d_%Y-Time_%H_%M_%S")
		self.fname = 'Data_' + self.date_label + '.npz'
		
		
		#start2goal_cm = [0,57] # cm
		#start2goal_ros = [-0.3,0.3]
		self.pos2dist = lambda _pos: (57./0.6)*(_pos+0.3)
		
		self.epi_data = []
		
		
		self.data = {}
		for name in self.names: self.data[name] = []
	def geti(self,name): return self.names.index(name)
	def get(self,name): return self.data[name]
	def add(self,name,value): self.data[name].append(value)
	def replace(self,name,value): self.data[name][-1] = value
	def reset(self):
		# export to perminant buffer
		#pckg = copy.deepcopy(self.epi_data_pckg)
		#for key in names: pckg[key] = self.data[name]
		#self.epi_data.append(copy.deepcopy(pckg))
		self.epi_data.append(copy.deepcopy(self.data))
		for name in self.nonsticky_names:
			self.data[name] = []
	def save(self):
		print(f'SAVING DATA AS [{self.fname}] [nEpi={len(self.epi_data)}]')
		np.savez_compressed(self.fname ,a=self.epi_data)
	def load(self,fname,trim=-1):
		plt.ioff()
		
		loaded = np.load(fname,allow_pickle=True)
		self.epi_data = loaded['a']
		
		data = self.epi_data[-2]
		print(f'LOADING DATA AS [{fname}]')
		print(f'\t| N episodes: {len(self.epi_data)}')
		print(f'\t| Data Types:')
		#for name in data.keys(): 
		for name in self.names: 
			is_missing = (name not in data.keys())
			if is_missing: print(f'\t\t -[MISSING] {name}')
			else: print(f'\t\t - {name}')
		

		
		
		T = data['Episode Length'][0]
		nsteps = len(data['Velocity'])	
		
		dt = T/len(data['Velocity'])
		t_epif = np.arange(nsteps)*dt
		
		#t_epif = np.arange(len(data['Velocity']))* (T/)
		NO_DATA = np.zeros(len(data['Velocity']))
		try: Ft_epif = data['Force']
		except: Ft_epif = np.zeros(len(data['Velocity']))

		speed = np.abs(np.array(data['Velocity']))
		begin_idx = np.array(np.where(speed>0.05)).flatten()[0]
		#_trim = np.array(np.where(speed[begin_idx:] < 0.1)).flatten()[0]
		#if len(_trim)>0:
		#	trim = _trim


		CumReward_epif = data['Episode Reward']
		Length_epif = np.array(data['Episode Length']) - dt*begin_idx
		try:
			Energy_epif = data['Episode Energy']
			Energy_epif[1:] = Energy_epif[1:] -  Energy_epif[:-2]
		except: Energy_epif = np.zeros(len(Length_epif))
		Dt_epif = data['Damping'][begin_idx:trim]
		vt_epif = data['Velocity'][begin_idx:trim]
		Ft_epif = Ft_epif[begin_idx:trim]
		t_epif = t_epif[begin_idx:trim]
		#Length_epif = t_epif[-1]
		
		
		# Calculation of Dynamics
		nsteps = len(vt_epif)
		vt_epif = np.array(vt_epif)		
		pt_epif = np.zeros(nsteps)
		for i in range(1,nsteps):
			# print(pt_epif.shape)
			# print(vt_epif.shape)
			#print(dt.shape)
			pt_epif[i] = pt_epif[i] + vt_epif[i-1]*dt
		at_epif = np.gradient(vt_epif)/dt
		Jt_epif = np.gradient(vt_epif)/dt
		xt_MJT,vt_MJT,at_MJT,Jt_MJT = get_MJT(tf = t_epif[-1],xf = pt_epif[-1],N= len(t_epif))
		
		
		
		stats = [CumReward_epif,Length_epif,Energy_epif]
		epif = [pt_epif,vt_epif,at_epif,Jt_epif,Dt_epif,Ft_epif,t_epif]
		MJT = [xt_MJT,vt_MJT,at_MJT,Jt_MJT]
		return stats,epif,MJT




def main():
	fnames = []
	Ntrims = []
	fname,Ntrim = 'YOUSEF_Data_-Date_11_17_2022-Time_18_55_40.npz',-20 #fname = 'Data_-Date_11_17_2022-Time_18_55_40.npz'
	fnames.append(fname)
	Ntrims.append(Ntrim)

	fname,Ntrim = 'NEERAJ_Data_-Date_11_17_2022-Time_19_40_51.npz',-45
	fnames.append(fname)
	Ntrims.append(Ntrim)

	fname,Ntrim = 'Data_-Date_11_18_2022-Time_15_41_44.npz',-10 # Vik
	fnames.append(fname)
	Ntrims.append(Ntrim)

	fname,Ntrim = 'Data_-Date_11_18_2022-Time_17_07_38.npz',-1 # mason
	fnames.append(fname)
	Ntrims.append(Ntrim)

	Ndata = len(fnames)
	for idata in range(Ndata):
		fname = fnames[idata]
		Ntrim = Ntrims[idata]


		Memory = MemoryHandler()
		stats,epif,MJT = Memory.load(fname,trim=Ntrim)
		CumReward_epif,Length_epif,Energy_epif = stats
		pt_epif,vt_epif,at_epif,Jt_epif,Dt_epif,Ft_epif,t_epif = epif
		xt_MJT, vt_MJT, at_MJT, Jt_MJT = get_MJT(tf=t_epif[-1], xf=57/10, N=len(t_epif))


		window = 10
		for i in range(len(t_epif)):
			win = window
			lb = int(max(i-win/2,0))
			ub = int(min(i+win/2,len(t_epif)))
			# print(f'{ub,lb,len(t_epif)}')
			Jt_epif[i] = np.mean(Jt_epif[lb:ub])
			Dt_epif[i] = np.mean(Dt_epif[lb:ub])

		nRows,nCols = 3,1
		fig_VJ,axs_VJ = plt.subplots(nRows,nCols)
		fig_VJ.set_size_inches(w=5, h=5)
		axs_VJ = np.array(axs_VJ).reshape([nRows,nCols])
		tnorm = np.linspace(0,1,len(vt_epif))
		axs_VJ[0, 0].plot(tnorm, vt_epif, label='OBS')
		axs_VJ[0, 0].plot(tnorm, vt_MJT,label='MJT')

		axs_VJ[1, 0].plot(tnorm, Dt_epif)
		axs_VJ[2, 0].plot(tnorm, Jt_epif)

		axs_VJ[0, 0].legend()


		#axs_VJ[0, 0].set_title(f'Subject {idata+1} Trajectory')
		axs_VJ[0, 0].set_ylabel('$V_t$')
		axs_VJ[1, 0].set_ylabel('$U_t$')
		axs_VJ[2, 0].set_ylabel('$J_t$')
		axs_VJ[-1, 0].set_xlabel('Normalized Time')
		for r in range(nRows): axs_VJ[r, 0].grid()
		plt.tight_layout()
		plt.savefig(f'plts/Fig_Subject{idata + 1}_TrajResults')




		nRows, nCols = 2, 1
		fig_EPI, axs_Epi = plt.subplots(nRows, nCols)
		fig_EPI.set_size_inches(w=5, h=3)
		axs_Epi = np.array(axs_Epi).reshape([nRows, nCols])

		tfit = np.arange(len(CumReward_epif))
		a, b = np.polyfit(tfit, CumReward_epif, 1)
		axs_Epi[0, 0].plot(CumReward_epif),axs_Epi[0, 0].set_ylabel('Reward')
		axs_Epi[0, 0].plot( a * tfit + b)

		a, b = np.polyfit(tfit, Length_epif, 1)
		axs_Epi[1, 0].plot(Length_epif), axs_Epi[1, 0].set_ylabel('Length')
		axs_Epi[1, 0].plot(a * tfit + b)
		#axs_Epi[2, 0].plot(Energy_epif), axs_Epi[2, 0].set_ylabel('Episode Energy')
		#axs_Epi[0, 0].set_title(f'Subject {idata + 1} Learning Results')
		for r in range(nRows): axs_Epi[r, 0].grid()
		axs_Epi[-1, 0].set_xlabel('Episode')
		# axs_VJ[0, 0].set_ylabel('$V(t)$')
		# axs_VJ[1, 0].set_ylabel('$J(t)$')
		# axs_VJ[2, 0].set_ylabel('$J(t)$')
		plt.tight_layout()
		plt.savefig(f'plts/Fig_Subject{idata + 1}_LearnResults')





	plt.show()
	
	nRows = 11
	nCols = 1
	
	fig,axs = plt.subplots(nRows,nCols)
	for i in [0,1,2]: axs[i].plot(stats[i])
	for i in range(4): axs[i+3].plot(epif[i])
	for i in range(4): axs[i+5].plot(MJT[i])
	plt.show()
	

if __name__ == "__main__":
	main()

#plt.show()
