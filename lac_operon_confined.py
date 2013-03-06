#!/usr/local/bin/python
from collections import namedtuple
from time import time
import timeit
from libsbml import *
import sys
import numpy as np
#from stocpu import Pnet
from build.src.stocuda import Pnet
import pyublas
#pyublas.set_trace(True)
np.set_printoptions(precision=10, suppress=True, linewidth=5000, threshold=10**6)


#CONSTANTS
#UNIVERSAL
LITER_PER_CUBIC_UM = 10.0**(-15.0)
NA = 6.022 * 10.0**23.0
D_OF_LAC = 380.0  #um^2/s
PARTICLES_PER_CUBIC_MICROMETER_PER_MICROMOLE = 602.2

#MODIFIABLE
CELL_REGION_RATIO = 10.0
CELL_NO = 10.0
MOLS_LAC = 13.0   #uM
DIFFUSION_BOUNDARY = 10.0   #um

#NOT VERY MODIFIABLE
REGION_RADIUS = (CELL_NO*CELL_REGION_RATIO)**(1.0/3.0)
REGION_SURFACE_AREA = 4.0*3.1415*REGION_RADIUS**2.0
REGION_VOLUME = (4.0/3.0)*3.1415*REGION_RADIUS**3.0
CELL_VOLUME = (4.0/3.0)*3.1415
PARTICLES_IN_REGION = PARTICLES_PER_CUBIC_MICROMETER_PER_MICROMOLE * REGION_VOLUME
PARTICLES_IN_CELL = PARTICLES_PER_CUBIC_MICROMETER_PER_MICROMOLE * CELL_VOLUME
DIFF_IN_CONSTANT = ((REGION_SURFACE_AREA * D_OF_LAC)/DIFFUSION_BOUNDARY) * MOLS_LAC
DIFF_OUT_CONSTANT = (REGION_SURFACE_AREA * D_OF_LAC)/DIFFUSION_BOUNDARY * (1.0/(REGION_VOLUME*LITER_PER_CUBIC_UM*NA))

def MakeMatrices():
    pass

def MakeInput():
    pass

class Parser(object):
    def __init__(self, fname):
        self.reader = SBMLReader()
        self.doc = self.reader.readSBML(fname)
        self.mod = self.doc.getModel()
        self.specs = self.Listtolist(self.mod.getListOfSpecies())
        self.specnames = [spec.getName() for spec in self.specs]
        self.specids = [spec.getId() for spec in self.specs]
        self.reacts = self.Listtolist(self.mod.getListOfReactions())
        self.matrixlist = ['P','T','Pre','Post','M','c']
        
        self.P = np.empty((len(self.specs),1), dtype='|S32')
        self.T = np.empty((len(self.reacts), 1), dtype='|S32')
        self.Pre = np.zeros((len(self.reacts), len(self.specs)), dtype=np.int32)
        self.Post = np.zeros((len(self.reacts), len(self.specs)), dtype=np.int32)
        self.M = np.zeros((len(self.specs),1), dtype=np.int32)
        self.c = np.zeros((len(self.reacts), 1), dtype=np.float32)
        
        for i, spec in enumerate(self.specs):
            self.P[i,0] = spec.getName()
            self.M[i,0] = spec.getInitialAmount()
        
        for i, react in enumerate(self.reacts):
            if react.getName() != '':
                self.T[i,0] = react.getName()
            else:
                self.T[i,0] = react.getId()
            for matrix, specrefs in zip([self.Pre, self.Post], [react.getListOfReactants(), react.getListOfProducts()]):
                for specref in specrefs:
                    matrix[i, self.specids.index(specref.getSpecies())] += specref.getStoichiometry()
            try:
                self.c[i,0] = react.getKineticLaw().getParameter(0).getValue()
            except AttributeError:
                found = False
                for j, kname in enumerate([param.getName() for param in self.mod.getListOfParameters()]):
                    try:
                        [z.strip() for y in [x.split() for x in react.getKineticLaw().getFormula().split('*')] for z in y].index(kname)
                        self.c[i,0] = self.mod.getListOfParameters().get(j).getValue()
                        found = True
                        break
                    except ValueError:
                        pass
                if found==False:
                    try:
                        self.c[i,0] = float(react.getKineticLaw().getFormula())
                    except ValueError:
                        self.c[i,0] = 0
#                 s = react.getKineticLaw().getFormula()
#                 j = [p.getName() for p in self.mod.getListOfParameters()].index(s)
#                 self.c[i,0] = self.mod.getListOfParameters().get(j).getValue()
    
    def PnetArgList(self):
        return (self.Pre, self.Post, self.M, self.c)
    
    def Replicate(self, nreacts=0, shared_species=(), addreactants=(), addproducts=(), addconstants=()):
        if nreacts > self.c.size:
            scale = nreacts/self.c.size
            pnetargs = []
            for i,ar in enumerate(self.PnetArgList()):
                arg = self.DilateArray(ar, scale)
                if shared_species and i!=3: #the 3 avoids touching the c matrix
                    arg = self.ShareSpecies(arg, scale, shared_species)
                pnetargs.append(arg)
            if addreactants and addproducts:
                pnetargs[0],pnetargs[1],pnetargs[3] = self.AddGlobalReaction(pnetargs[0],pnetargs[1],pnetargs[3], reactants=addreactants, products=addproducts, constants=addconstants)
            return pnetargs
        else:
            return self.PnetArgList()
    
    def ShareSpecies(self, mat, scale, shared_species):
        eliminate = []
        for i in  range(1, scale):
            for species in shared_species:
                eliminate.append(self.M.size*i + species)
        if mat.shape[1]==1:
            for i in reversed(eliminate):
                mat = np.delete(mat, i, axis=0)
        else:
            for i,j in zip(reversed(eliminate), reversed(shared_species*(scale-1))):
                mat[:,j]+=mat[:,i]
                mat = np.delete(mat, i, axis=1)
        return mat
    
    def __str__(self):
        s = []
        for mat in self.matrixlist:
            s += [mat, ' =\n', self.__getattribute__(mat).__str__(), '\n']
        return ''.join(s)
    
    @staticmethod
    def AddGlobalReaction(Pre, Post, c, reactants, products, constants):
        ret = []
        for mat, speclists in zip ((Pre,Post),(reactants,products)):
            for speclist in speclists:
                tmp = [0]*Pre.shape[1]
                for spec in speclist:
                    tmp[spec]+=1
                mat = np.vstack((mat, np.array(tmp, dtype=mat.dtype)))
            ret.append(mat)
        for constant in constants:
            c = np.vstack((c, np.array(constant, dtype=c.dtype)))
        ret.append(c)
        return ret
    
    @staticmethod
    def DilateArray(ar, scale):
        if ar.shape[1]==1:
            return np.bmat([[ar]]*scale)
        else:
            background = np.array([[np.zeros(ar.shape)]*scale]*scale, dtype=ar.dtype)
            for i in xrange(scale):
                background[i,i] = ar
            return np.array(np.bmat(background.tolist()), dtype=np.int32)
            #real = background.view('int32')[:,::2]
            #real[:] = background
            #return real          
            
    @staticmethod
    def Listtolist(List):
        l = []
        for i in range(List.size()):
            l.append(List.get(i))
        return l

class Timer(object):
    iheaders = ['enlarge', 'step', 'repeat', 'fpath']
    
    def __init__(self, fpath, enlarge, step, repeat):
        self.fpath = fpath
        self.enlarge = enlarge
        self.step = step
        self.repeat = repeat
        self.dilationts = []
        self.pnetsetupts = []
        self.gillespiets = []
        
    def Go(self):
        for i in range(self.repeat):
            for t, ts in zip(self.Time(), self.GetTs()):
                ts.append(t)
        
    def Time(self):
        parser = Parser(self.fpath)
        now = time()
        args = parser.Enlarge(self.enlarge)
        dilationt = time() - now
        now = time()
        #print args
        pnet = Pnet(*args)
        pnetsetupt = time() - now
        now = time()
        pnet.Gillespie(10000)
        gillespiet = time() - now
        return dilationt, pnetsetupt, gillespiet
    
    def GetInput(self):
        return [self.__getattribute__(s) for s in self.__class__.iheaders]
    
    def GetTs(self, mean=False):
        if mean:
            return [np.mean(ts) for ts in self._GetTs()]
        else:
            return self._GetTs()
            
    def _GetTs(self, mean=False):
        return self.dilationts, self.pnetsetupts, self.gillespiets
    
    def __str__(self):
        out = ''
        rheaders = ['reaction_network_dilation', 'pnet_setup', 'gillespie']
        for l in (self.__class__.iheaders, self.GetInput(), rheaders, self.GetTs(mean=True)):
            out += '\t'.join([str(s) for s in l]) + '\n'
        return out
    
def debug():
    parser = Parser('/home/likewise-open/WIN/cklein13/models/BIOMD0000000091.xml')#Parser(sys.argv[1])
    now = time()
    #pnet = Pnet(*parser.PnetArgList())
    args = parser.Replicate(200)
    print 'reaction network dilation took %.3f seconds' % (time() - now)
    now = time()
    pnet = Pnet(*args)
    print 'pnet setup took %.3f seconds' % (time() - now)
    now = time()
    pnet.Gillespie(10000)
    print 'this took %.3f seconds' % (time() - now)
    #print parser

if __name__=='__main__':
    fpath = sys.argv[1]
    parser = Parser(fpath)
    args = parser.Replicate(46, (11,), [[11],[]], [[],[11]], [DIFF_OUT_CONSTANT,DIFF_IN_CONSTANT])
    args[2][10:12,:] = np.array([[PARTICLES_IN_CELL],[PARTICLES_IN_REGION]])
    for arg in args:
        print arg.dtype
    pnet = Pnet(*args)
    pnet.Gillespie(10000)
#     repeat = 10
#     TimerInput = namedtuple('TimerInput', Timer.iheaders)
#     smalltimerinput  = TimerInput(fpath=fpath, enlarge=100, step=10000, repeat=repeat)
#     mediumtimerinput = TimerInput(fpath=fpath, enlarge=1000, step=10000, repeat=repeat)
#     largetimerinput  = TimerInput(fpath=fpath, enlarge=10000, step=10000, repeat=repeat)
#     #timers = [Timer(**smalltimerinput._asdict()), Timer(**mediumtimerinput._asdict()), Timer(**largetimerinput._asdict())]
#     timers = [Timer(**smalltimerinput._asdict())]
#     for timer in timers:
#         timer.Go()
#         print timer