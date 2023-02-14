"""    This script only designed for evolution on the surface."""import numpy as npimport matplotlib.pyplot as pltfrom mpl_toolkits.mplot3d import Axes3Ddef distance(a, b) -> float:	return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)class Surface:	"""		Infinite surface of the growth， initalized by height=0	"""	def __init__(self, width):		self.width = width		self.area = np.zeros((width, width))		self.grow_comp = np.ones((width, width))		self.bridge_comp = np.ones((width, width))		self.every_growth = 1.0		self.miu = 0		self.sigma = 16		self.sigma_inhibt = 8		self.comp_func = lambda x: 2 * np.exp(-(x - self.miu) ** 2 / (2 * self.sigma ** 2)) / (np.sqrt(2*np.pi)*self.sigma)		self.inhabit_func = lambda x: -1 * 1 * np.exp(-(x - self.miu) ** 2 / (2 * self.sigma_inhibt ** 2)) / (np.sqrt(2*np.pi)*self.sigma_inhibt)		self.iter = 0		self.au_active_dist = 180		self.ligand_limit_dist = 80		self.au_speed = 8		self.au_list = []	def view_2d(self, title=""):		"""			plot of the surface.		"""		plt.matshow(self.grow_comp)		# fig, axs = plt.subplots(2, 1)		# axs[0].matshow(self.area)		# axs[1].matshow(self.grow_comp)		plt.title(title)		plt.colorbar()		plt.show()			def view_3d(self, title=""):		"""			plot of the surfce with 3d mode.		"""		fig = plt.figure()		ax = fig.gca(projection='3d')		x = np.linspace(0, 10, self.width)		y = np.linspace(0, 10, self.width)		X, Y = np.meshgrid(x, y)		Z = self.area		ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)		if np.max(Z) < 100:			ax.set_zlim(0, 100)		elif np.max(Z) < 1000:			ax.set_zlim(0, 1000)		else:			ax.set_zlim(0, 20000)		ax.spines['top'].set_visible(False)		ax.spines['bottom'].set_visible(False)		ax.spines['left'].set_visible(False)		ax.spines['right'].set_visible(False)		ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))		ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))		ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))		plt.axis('off')		plt.title(title)		plt.show()		def evolute(self):		"""			Evolution process with total morphology.		"""		curr_state = np.zeros((self.width, self.width))		au_locs = []						# Au 沉积位置		for _ in range(self.au_speed):			x_ = np.random.randint(0, self.width)			y_ = np.random.randint(0, self.width)			au_locs.append((x_, y_))			self.au_list.append((x_, y_))		# Au 沉积点		for loc in au_locs:			curr_state[loc[0], loc[1]] += (self.grow_comp[loc[0], loc[1]]) * self.every_growth			self.grow_comp[loc[0], loc[1]] += self.comp_func(self.area[loc[0], loc[1]])			# Au 沉积点附近			for i in range(loc[0]-self.au_active_dist, loc[0]+self.au_active_dist):				for j in range(loc[1]-self.au_active_dist, loc[1]+self.au_active_dist):					_distance_ = np.sqrt((i - loc[0]) ** 2 + (j - loc[1]) ** 2)					if _distance_ > 0 and (self.width > i > 0) and (0 < j < self.width):						if _distance_ < self.au_active_dist:							self.grow_comp[i, j] += self.comp_func(_distance_ * self.grow_comp[i, j] * 1.2)			# Au bridge			for m, n in self.au_list:				dist_2p = distance((loc[0], loc[1]), (m, n))				if dist_2p < 50:					A = n - loc[1]					B = loc[0] - m					C = loc[0] * (loc[1]-n) + loc[1] * (m-loc[0])					for i in range(loc[0], m):						for j in range(loc[1], n):							dist_p2l = np.abs(A*i + B*j + C) / np.sqrt(A**2 + B**2)							if dist_p2l < 0.1 * dist_2p:								# self.bridge_comp[i, j] += self.bridge_comp[i, j] * (2 + np.exp(-1 * dist_p2l))								self.bridge_comp[i, j] += self.comp_func(dist_p2l)						# Ligand 落地位置		if np.random.randn() < 0.6 and self.iter > 100 and False:			x, y = np.argmin(self.area)		else:			x = np.random.randint(0, self.width)			y = np.random.randint(0, self.width)		# self.grow_comp[x, y] = 0		# Ligand 落地位置附近		for i in range(x-self.ligand_limit_dist, x+self.ligand_limit_dist):			for j in range(y-self.ligand_limit_dist, y+self.ligand_limit_dist):				_distance_ = np.sqrt((i - x) ** 2 + (j - y) ** 2)				if _distance_ > 0 and (self.width > i > 0) and (0 < j < self.width):					if _distance_ < self.ligand_limit_dist / 3:						self.grow_comp[i, j] += self.inhabit_func(_distance_)						self.area += curr_state + self.bridge_comp		self.area = 1 / (1+np.exp(-1 * self.area))		# self.area = (self.area - np.mean(self.area)) / (np.max(self.area) - np.min(self.area))		self.iter += 1if __name__ == '__main__':	surface = Surface(200)	surface.view_2d("Initalized State")	for _ in range(2000):		surface.evolute()		if _ % 10 == 0:			surface.view_2d(f"Evolution: {_}")			