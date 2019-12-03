
import networkx as nx
import numpy as np

from scipy.optimize import minimize
from scipy.spatial.distance import cdist

epsilon=0.0001

RECTANGLE = 0x0010
ASYMETRY  = 0x0100



def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def convert_angles(a):
	while(np.any(a<=-np.pi)):
		a[a<=-np.pi] += 2*np.pi
	while(np.any(a>np.pi)):
		a[a>np.pi] -= 2*np.pi

def convert_angle(a):
	while(a<=-np.pi):
		a += 2*np.pi
	while(a>np.pi):
		a -= 2*np.pi
	return a

def convert_angles_pos(a):
	while(np.any(a<0)):
		a[a<0] += 2*np.pi
	while(np.any(a>2*np.pi)):
		a[a>2*np.pi] -= 2*np.pi

def paramsAsMatrices(params):
	u1, u2, alpha, l1, l2, a1, a2, p = params
	u = np.matrix([u1, u2]) 
	R = np.matrix([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])
	S = np.matrix([[l1,0],[0,l2]])
	A = np.matrix([[a1,0],[0,a2]])
	T = R*S
	TI=T.I
	return u, TI, A, p

	
def dG(a, b, TI, A, q):
	try:
		np.seterr(over= 'warn',under= 'warn')
		x = np.matrix(b-a).T
		z = TI*x
		AxI=np.matrix(np.diag(np.exp((np.flip(-z.T, axis=0)*A).A[0])))
		res = np.power(np.abs(np.sum(AxI*np.power(np.abs(z),q), axis=0)),1.0/q).A
	except BaseException:
		print('utils.py - dG() - BaseException')
		import pdb
		pdb.set_trace()
	return res 


def fit_all(Z, label,distance_type=''): 
	#distance_type = 'all'#'p_only'#'A_only' #''

	if len(distance_type)==1 :
		pNum = int(distance_type)
		if pNum == 5:
			distance_type = ''
		elif pNum == 6:
			distance_type = 'A_only'
		elif pNum == 7:
			distance_type = 'p_only'
		elif pNum == 8:
			distance_type = 'all'
		else : 
			print(' parameters number not understood')

	def err(params, X):
		u1, u2, alpha, l1, l2, a1, a2, p = params
		u, TI, A, p = paramsAsMatrices(params)
		return np.mean([(dG(u.A[0], x.A[0], TI, A, p) - 1)**2 for x in X]) * np.linalg.norm([u1, u2, l1, l2])



	Z100 = np.matrix(np.unique(_create_sorted_contour(Z, center=np.mean(Z,axis=0), rot=0, N=100, display=False), axis=0))

	m  = np.mean(Z100.A, axis=0)
	_Z100 = (Z100 - m)
	u1, u2 = 0, 0
	Cx = (_Z100.T*_Z100)/_Z100.shape[0]	
	Lx, Rx = np.linalg.eigh(Cx)
	if np.any(Lx == 0):
		Lx = Lx + epsilon	

	alpha = -np.arctan2(Rx[0,1], Rx[1,1])
	a1, a2 = 0, 0
	p = 2

	l1, l2 = np.sqrt(2*Lx)


	
	if distance_type is 'p_only':
		try:
			def p_err(params, X):
				u1, u2, alpha, l1, l2, p = params
				a1, a2 = 0, 0
				return err((u1, u2, alpha, l1, l2, a1, a2, p), X)
			min_res = minimize(p_err, x0=(u1,u2,alpha,l1,l2,p), args=(_Z100,), 
										bounds=((None,None), 
										(None,None), 
										(-np.pi,np.pi), 
										#(0,None), 
										#(0,None), 
										(l1-0.5*l1,l1+0.5*l1), 
										(l2-0.5*l2,l2+0.5*l2),
										(1,15))) 
			
			if min_res.success: 
				u1, u2, alpha, l1, l2, p = min_res.x
			else:
				print('optimization did not work : %s' % lab)
		except BaseException:
			print('utils.py - fit_all() - BaseException')
			print(label)

	elif distance_type is 'A_only':
		try:
			def A_err(params, X):
				u1, u2, alpha, l1, l2, a1, a2 = params
				p = 2
				return err((u1, u2, alpha, l1, l2, a1, a2, p), X)
			min_res = minimize(A_err, x0=(u1,u2,alpha,l1,l2,a1,a2), args=(_Z100,), 
										bounds=((None,None), 
										(None,None), 
										(-np.pi,np.pi), 
										(l1-0.5*l1,l1+0.5*l1), 
										(l2-0.5*l2,l2+0.5*l2),
										(-0.5,0.5), 
										(-0.5,0.5))) 
			
			if min_res.success: 
				u1, u2, alpha, l1, l2, a1, a2 = min_res.x
			else:
				print('optimization did not work : %s' % lab)
		except BaseException:
			print('utils.py - fit_all() - BaseException')
			print(label)
	
	elif distance_type is 'all':
		try:
			min_res = minimize(err, x0=(u1,u2,alpha,l1,l2,a1,a2,p), args=(_Z100,), 
										bounds=((None,None), 
										(None,None), 
										(-np.pi,np.pi), 
										(l1-0.5*l1,l1+0.5*l1), 
										(l2-0.5*l2,l2+0.5*l2),
										(-0.5,0.5), 
										(-0.5,0.5),  
										(1,15))) 
			
			if min_res.success: 
				u1, u2, alpha, l1, l2, a1, a2, p = min_res.x
			else:
				print('optimization did not work : %s' % lab)
		except BaseException:
			print('utils.py - fit_all() - BaseException')
			print(label)
	else : 
		try:
			def ellipse_err(params, X):
				u1, u2, alpha, l1, l2, a1, a2 = params
				p = 2
				a1, a2 = 0, 0
				return err((u1, u2, alpha, l1, l2, a1, a2, p), X)
			min_res = minimize(ellipse_err, x0=(u1,u2,alpha,l1,l2,a1,a2), args=(_Z100,), 
										bounds=((None,None), 
										(None,None), 
										(-np.pi,np.pi), 
										(l1-0.5*l1,l1+0.5*l1), 
										(l2-0.5*l2,l2+0.5*l2),
										(-0.5,0.5), 
										(-0.5,0.5))) 
			
			if min_res.success: 
				u1, u2, alpha, l1, l2, a1, a2 = min_res.x
			else:
				print('optimization did not work : %s' % lab)
		except BaseException:
			print('utils.py - fit_all() - BaseException')
			print(label)
	u1 = u1 + m[0] 
	u2 = u2 + m[1]
		
	return u1, u2, alpha, l1, l2, a1, a2, p


def fit_loc_angle(Z, params, params_orig, lab):  
	
	def err(params, X, l1, l2, a1, a2, p):
		u1, u2, alpha = params
		u, TI, A, p = paramsAsMatrices(np.concatenate((params, (l1, l2, a1, a2, p))))
		return np.mean([(dG(u.A[0], x.A[0], TI, A, p) - 1)**2 for x in X]) * np.linalg.norm([u1, u2, l1, l2, a1*a2])

	u1, u2, alpha = params
	u10, u20, alpha0, l10, l20, a10, a20, p0 = params_orig

	Z100 = np.matrix(np.unique(_create_sorted_contour(Z, center=np.mean(Z,axis=0), rot=0, N=100, display=False), axis=0))
	m  = np.mean(Z100.A, axis=0)
	_Z100 = (Z100 - m)
	u1, u2 = 0, 0
	
	try :
		np.seterr(all= 'warn')
		min_res = minimize(err, x0=(u1,u2,alpha), args=(_Z100, l10, l20, a10, a20, p0),bounds=((None,None),(None,None),(alpha-np.pi,alpha+np.pi)))
	except BaseException:
		print('utils.py - fit_loc_angle() - BaseException')
		import pdb
		pdb.set_trace()
								
	if min_res.success:
		u1, u2, alpha = min_res.x
	else:
		print('optimization did not work : %s' % lab)

	u1 = u1 + m[0]
	u2 = u2 + m[1]
	
	return u1, u2, alpha

	
	
	
def _create_sorted_contour(yxcontour, center=None, rot=0, N=100, display=False):
	try:
		
		cc = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]
		
		# nearest neighbor graph
		import networkx as nx
		dist = cdist(yxcontour,yxcontour)	
		dist[dist >= 2] = 0
		G = nx.from_numpy_matrix(dist)
				
		# in rare cases there can be several subgraph, the largest is then selected
		if nx.number_connected_components(G) > 1:
			GG = [G.subgraph(c) for c in nx.connected_components(G)]
			G = GG[np.argmax([nx.number_of_nodes(g) for g in GG])]
		

		# Let's choose n1->n2, the first edge of the contour
		# n1 should be in G and the most south-left point 

		Gnodes = np.array(G.nodes())

		miny_idx = np.where(yxcontour[Gnodes,0] == np.min(yxcontour[Gnodes,0]))[0]
		n1 = Gnodes[miny_idx][np.argmin(yxcontour[Gnodes][miny_idx,1])]
		
		# n2 is the first neighbor of n1 after direct rotation from [-1,-1], to ensure tracing the outside contour
		CC = [list(a) for a in np.roll(cc, -cc.index([-1,-1]), axis=0)[1:]]
		V = [list(yxcontour[ni] - yxcontour[n1]) for ni in G.neighbors(n1)]
		n2 = list(G.neighbors(n1))[np.argmin([CC.index(vi) for vi in V])]

		nfirst = n1
		nsec   = n2
		nnnyx  = yxcontour[nfirst] - 1
		
		G2 = nx.DiGraph() # directed graph
		G2.add_node(n1)
		G2.add_node(n2)
		G2.add_edge(n1,n2)
		while(n2 != nfirst): 
			neigh = list(G.neighbors(n2))
			neigh.remove(n1)
			
			if len(neigh) == 0:
				tmp=n2
				n2=n1
				n1=tmp
			elif len(neigh) == 1:
				n1=n2
				n2=neigh[0]
			else:
				v = list(yxcontour[n1] - yxcontour[n2])
				CC = [list(a) for a in np.roll(cc, -cc.index(v), axis=0)[1:]]
				V = [list(yxcontour[ni] - yxcontour[n2]) for ni in neigh]
				n1 = n2
				n2 = neigh[np.argmin([CC.index(vi) for vi in V])]
			G2.add_node(n1)
			G2.add_node(n2)
			G2.add_edge(n1,n2)

		# in rare cases there can be more than a cycle (=for eg when two cycles connected by a single node), the largest is kept
		cycles = tuple(nx.simple_cycles(G2))
		G3 = nx.DiGraph()
		nx.add_cycle(G3, cycles[np.argmax([len(cy) for cy in cycles])])
		
		# Selection of nstart in G3: the closest angle to rot given the center		
		rot = convert_angle(rot) # to make sure it is between -pi and +pi
		G3nodes = np.array(G3.nodes())
		trcontour = np.vstack(cart2pol(yxcontour[G3nodes,1] - center[1], yxcontour[G3nodes,0] - center[0])).T
		idxSortedcontour = np.argsort(trcontour[:,0])
		_idx = np.searchsorted(trcontour[:,0], rot, sorter=idxSortedcontour)
		nstart = G3nodes[idxSortedcontour][_idx if _idx<idxSortedcontour.shape[0] else 0]

		path  = np.array(nx.find_cycle(G3, source=nstart)) # contour starting from the rotated init
		pathd = [dist[u,v] for (u,v) in path]
		cumsum = np.copy(pathd)
		
		for k in range(1,cumsum.shape[0]):
			cumsum[k] += cumsum[k-1]
		
		short_contour_idx = path[:,0][np.searchsorted(cumsum, np.arange(N)*cumsum[-1]/N)]

		if display:
			import matplotlib.pyplot as plt
			fig = plt.figure()
			ax = fig.add_subplot(1, 1, 1)

			G4 = nx.Graph()
			nodes = list(range(short_contour_idx.shape[0]))
			G4.add_nodes_from(nodes)
			G4.add_edges_from(np.array((nodes,np.roll(nodes, 1))).T)
			
			nx.draw(G, yxcontour, node_size=20, node_color ='k', edge_color='k')
			nx.draw(G3, yxcontour, node_size=20, node_color ='y', edge_color='y', with_labels=True)
			nx.draw(G4, yxcontour[short_contour_idx], node_size=20, node_color ='g', edge_color='g')
			plt.plot(yxcontour[0,0], yxcontour[0,1],'ob')
			plt.plot(yxcontour[nfirst,0], yxcontour[nfirst,1],'dg',ms=20)
			plt.plot(yxcontour[nsec,0], yxcontour[nsec,1],'Dg',ms=20)
			plt.plot(nnnyx[0], nnnyx[1],'or',ms=20)
			plt.plot(yxcontour[path[0,0],0], yxcontour[path[0,0],1],'or')
			plt.axis('equal')
			plt.show()
			plt.close()
			import pdb
			pdb.set_trace()

		return yxcontour[short_contour_idx]
	except nx.NetworkXError as e:
		print('utils.py - _create_sorted_contour() - nx.NetworkXError: %s' % e)
		return None
	except ValueError as e:
		print('utils.py - _create_sorted_contour() - ValueError : %s' % e)
		import pdb
		pdb.set_trace()
		return None
	except IndexError as e: 
		print('utils.py - _create_sorted_contour() - IndexError : %s' % e)
		return None
	except MemoryError as e: 
		print('utils.py - _create_sorted_contour() - MemoryError : %s' % e)
		import pdb
		pdb.set_trace()
		return None

