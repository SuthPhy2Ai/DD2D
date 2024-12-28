import numpy as np
from ase.io import read, write
from ase.atoms import Atoms
from ase.db import connect
db = connect("test.db")
def trans2vasp(txtpath):
    data = np.loadtxt(txtpath)
    a,b,c =  data[:3]
    alpha,beta,gamma = data[3:6]
    # gamma = 90
    pos_ = data[6:]
    pos_elem = pos_[3::4]
    pos_x = pos_[0::4]
    pos_y = pos_[1::4]
    pos_z = pos_[2::4]
    positons = np.array([pos_x,pos_y,pos_z]).T
    atoms_ = Atoms(positions=positons, symbols=pos_elem)
    atoms_.set_cell(atoms_.cell.fromcellpar([a,b,c,alpha,beta,gamma]))
    file_tag = txtpath.split(".")[0]
    write(f"{file_tag}.vasp",atoms_)
    db.write(atoms_,txt=txtpath)
import os

for file in os.listdir():
    if file.endswith(".txt"):
        trans2vasp(file)
