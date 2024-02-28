import os
from pathlib import Path
from typing import Dict, Optional, Any, List
import json


def _fe_var_check(var, varname):
    msg = f"{varname} has to be set when run free energy simulation"
    if isinstance(var, str):
        assert var != "", msg
    else:
        assert (var is not None), msg


def pmemd_exec(use_cuda: bool = True, use_mpi: bool = False):
    if use_cuda:
        exe = "pmemd.cuda"
    else:
        exe = "pmemd"
    return exe


def pmemd_command(pmemed_exec, prmtop, inpcrd, deffnm):
    with open(Path(__file__).parent / 'run.sh', 'r') as f:
        command = f.read()
    return command.format(prmtop=prmtop, inpcrd=inpcrd, deffnm=deffnm, pmemd_exec=pmemed_exec)


def em(
    wdir: os.PathLike,
    prmtop: os.PathLike,
    inpcrd: os.PathLike,
    pmemd_exec: str = "pmemd.cuda",
    num_steps: int = 5000,
    ofreq: Optional[int] = None,
    cutoff: float = 10.0,
    free_energy: bool = True,
    clambda: Optional[float] = None,
    noshakemask: str = "",
    timask1: str = "",
    timask2: str = "",
    scmask1: str = "",
    scmask2: str = "",
    deffnm: str = "em",
):
    """
    Energy minimization
    """
    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)
    prmtop = Path(prmtop).resolve()
    inpcrd = Path(inpcrd).resolve()
    with open(Path(__file__).parent / 'em.in') as f:
        template = f.read()
    ofreq = int(num_steps // 10) if ofreq is None else ofreq
    
    if free_energy:
        _fe_var_check(noshakemask, "noshakemask")
        _fe_var_check(timask1, "timask1")
        _fe_var_check(timask2, "timask2")
        _fe_var_check(scmask1, "scmask1")
        _fe_var_check(scmask2, "scmask2")
        _fe_var_check(clambda, 'clambda')
        ifsc, icfe = 1, 1
    else:
        ifsc, icfe = 0, 0
        clambda = 0.0

    inpstr = template.format(
        maxcyc=num_steps, ofreq=ofreq,
        cut=cutoff, 
        ifsc=ifsc, icfe=icfe,
        clambda=clambda,
        gti_cut_sc_on=cutoff - 2.0, gti_cut_sc_off=cutoff,
        noshakemask=noshakemask, timask1=timask1, timask2=timask2,
        scmask1=scmask1, scmask2=scmask2
    )
    with open(wdir / f'{deffnm}.in', 'w') as f:
        f.write(inpstr)
    
    cmdstr = pmemd_command(pmemd_exec, prmtop, inpcrd, deffnm)
    with open(wdir / f'{deffnm}.sh', 'w') as f:
        f.write(cmdstr)



def heat(
    wdir: os.PathLike,
    prmtop: os.PathLike,
    inpcrd: os.PathLike,
    pmemd_exec: str = "pmemd.cuda",
    num_steps: int = 5000,
    ofreq: Optional[int] = None,
    dt: float = 0.001,
    temp0: float = 298.15,
    tempi: float = 0.0,
    restraint_wt: float = 5.0,
    cutoff: float = 10.0,
    free_energy: bool = True,
    clambda: Optional[float] = None,
    noshakemask: str = "",
    timask1: str = "",
    timask2: str = "",
    scmask1: str = "",
    scmask2: str = "",
    deffnm: str = 'heat'
):
    """
    Energy minimization
    """
    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)
    prmtop = Path(prmtop).resolve()
    inpcrd = Path(inpcrd).resolve()
    with open(Path(__file__).parent / 'heat.in') as f:
        template = f.read()
    ofreq = int(num_steps // 10) if ofreq is None else ofreq
    
    if free_energy:
        _fe_var_check(noshakemask, "noshakemask")
        _fe_var_check(timask1, "timask1")
        _fe_var_check(timask2, "timask2")
        _fe_var_check(scmask1, "scmask1")
        _fe_var_check(scmask2, "scmask2")
        _fe_var_check(clambda, 'clambda')
        ifsc, icfe = 1, 1
        ntf = 1
    else:
        ifsc, icfe = 0, 0
        ntf = 2
    
    ntr = 1 if restraint_wt != 0 else 0

    inpstr = template.format(
        nstlim=num_steps, ofreq=ofreq, dt=dt,
        ntr=ntr, restraint_wt=restraint_wt,
        cut=cutoff, 
        temp0=temp0, tempi=tempi,
        ifsc=ifsc, icfe=icfe,
        clambda=clambda,
        gti_cut_sc_on=cutoff - 2.0, gti_cut_sc_off=cutoff,
        istep2=int(num_steps // 2),
        ntf=ntf,
        noshakemask=noshakemask, timask1=timask1, timask2=timask2,
        scmask1=scmask1, scmask2=scmask2
    )
    with open(wdir / f'{deffnm}.in', 'w') as f:
        f.write(inpstr)
    
    cmdstr = pmemd_command(pmemd_exec, prmtop, inpcrd, deffnm)
    with open(wdir / f'{deffnm}.sh', 'w') as f:
        f.write(cmdstr)


def pressurize(
    wdir: os.PathLike,
    prmtop: os.PathLike,
    inpcrd: os.PathLike,
    pmemd_exec: str = "pmemd.cuda",
    num_steps: int = 5000,
    ofreq: Optional[int] = None,
    dt: float = 0.001,
    temp0: float = 298.15,
    pressure: float = 1.01325,
    restraint_wt: float = 5.0,
    cutoff: float = 10.0,
    free_energy: bool = True,
    clambda: Optional[float] = None,
    noshakemask: str = "",
    timask1: str = "",
    timask2: str = "",
    scmask1: str = "",
    scmask2: str = "",
    deffnm: str = 'pres_0'
):
    """
    Energy minimization
    """
    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)
    prmtop = Path(prmtop).resolve()
    inpcrd = Path(inpcrd).resolve()
    with open(Path(__file__).parent / 'pres.in') as f:
        template = f.read()
    ofreq = int(num_steps // 10) if ofreq is None else ofreq
    
    if free_energy:
        _fe_var_check(noshakemask, "noshakemask")
        _fe_var_check(timask1, "timask1")
        _fe_var_check(timask2, "timask2")
        _fe_var_check(scmask1, "scmask1")
        _fe_var_check(scmask2, "scmask2")
        _fe_var_check(clambda, 'clambda')
        ifsc, icfe = 1, 1
        ntf = 1
    else:
        ifsc, icfe = 0, 0
        ntf = 2
    
    ntr = 1 if restraint_wt != 0 else 0

    inpstr = template.format(
        nstlim=num_steps, ofreq=ofreq, dt=dt,
        ntr=ntr, restraint_wt=restraint_wt,
        cut=cutoff, 
        temp0=temp0,
        ifsc=ifsc, icfe=icfe,
        clambda=clambda,
        gti_cut_sc_on=cutoff - 2.0, gti_cut_sc_off=cutoff,
        ntf=ntf,
        pres0=pressure,
        noshakemask=noshakemask, timask1=timask1, timask2=timask2,
        scmask1=scmask1, scmask2=scmask2
    )
    with open(wdir / f'{deffnm}.in', 'w') as f:
        f.write(inpstr)
    
    cmdstr = pmemd_command(pmemd_exec, prmtop, inpcrd, deffnm)
    with open(wdir / f'{deffnm}.sh', 'w') as f:
        f.write(cmdstr)


def prod(
    wdir: os.PathLike,
    prmtop: os.PathLike,
    inpcrd: os.PathLike,
    pmemd_exec: str = "pmemd.cuda",
    num_steps: int = 5000,
    ofreq: Optional[int] = None,
    dt: float = 0.001,
    temp0: float = 298.15,
    pressure: float = 1.01325,
    restraint_wt: float = 0.0,
    cutoff: float = 10.0,
    free_energy: bool = True,
    clambda: Optional[float] = None,
    use_mbar: bool = True,
    lambdas: Optional[List[float]] = None,
    efreq: Optional[int] = None,
    numexchg: Optional[int] = None,
    noshakemask: str = "",
    timask1: str = "",
    timask2: str = "",
    scmask1: str = "",
    scmask2: str = "",
    deffnm: str = 'prod'
):
    """
    Energy minimization
    """
    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)
    prmtop = Path(prmtop).resolve()
    inpcrd = Path(inpcrd).resolve()
    with open(Path(__file__).parent / 'prod.in') as f:
        template = f.read()
    ofreq = int(num_steps // 10) if ofreq is None else ofreq
    
    if free_energy:
        _fe_var_check(noshakemask, "noshakemask")
        _fe_var_check(timask1, "timask1")
        _fe_var_check(timask2, "timask2")
        _fe_var_check(scmask1, "scmask1")
        _fe_var_check(scmask2, "scmask2")
        _fe_var_check(clambda, 'clambda')
        ifsc, icfe = 1, 1
        ntf = 1
    else:
        ifsc, icfe = 0, 0
        ntf = 2
    
    ntr = 1 if restraint_wt != 0 else 0

    if use_mbar:
        _fe_var_check(lambdas, "lambdas")
        _fe_var_check(efreq, "efreq")
        mbar_setting = [
            "{:<15} = 1".format("ifmbar"), 
            "{:<15} = {}".format("bar_intervall", efreq),
            "{:<15} = {}".format("mbar_states", len(lambdas)),
            "{:<15} = {}".format("mbar_lambda", ",".join(str(x) for x in lambdas))
        ]
        mbar_setting = "\n".join(mbar_setting)
    else:
        efreq = ofreq if efreq else efreq
        mbar_setting = ""
        
    inpstr = template.format(
        nstlim=num_steps, ofreq=ofreq, dt=dt,
        ntr=ntr, restraint_wt=restraint_wt,
        cut=cutoff, 
        temp0=temp0,
        ifsc=ifsc, icfe=icfe,
        clambda=clambda,
        gti_cut_sc_on=cutoff - 2.0, gti_cut_sc_off=cutoff,
        ntf=ntf,
        pres0=pressure,
        noshakemask=noshakemask, timask1=timask1, timask2=timask2,
        scmask1=scmask1, scmask2=scmask2,
        numexchg=numexchg, mbar_setting=mbar_setting,
        efreq=efreq
    )
    with open(wdir / f'{deffnm}.in', 'w') as f:
        f.write(inpstr)


def fep_workflow(config, wdir):
    lambdas = config['lambdas']
    inpcrd = config['inpcrd']
    prmtop = config['prmtop']

    mask_config = {
        key: config[key] for key in ['noshakemask', 'timask1', 'timask2', 'scmask1', 'scmask2']
    }

    pmemd_exec = 'pmemd.cuda'

    wdir = Path(wdir).resolve()
    wdir.mkdir(exist_ok=True)

    with open(Path(__file__).parent / 'default_settings.json') as f:
        defaults = json.load(f)
    
    temp = config.get("temperature", defaults['temperature'])
    pres = config.get("pressure", defaults['pressure'])
    cutoff = config.get("cutoff", defaults['cutoff'])
    

    for i, clambda in enumerate(lambdas):
        lambda_dir = wdir / f"lambda{i}"
        lambda_dir.mkdir(exist_ok=True)
        
        em_dir = lambda_dir / "em"
        defaults['em'].update(config.get('em', {}))
        em(
            wdir=em_dir,
            prmtop=prmtop, inpcrd=inpcrd,
            pmemd_exec=pmemd_exec,
            free_energy=True,
            cutoff=cutoff,
            clambda=clambda,
            deffnm="em",
            **defaults['em'],
            **mask_config,
        )

        heat_dir = lambda_dir / 'heat'
        defaults['heat'].update(config.get('heat', {}))
        heat(
            wdir=heat_dir,
            prmtop=prmtop,
            inpcrd=em_dir / "em.rst7",
            pmemd_exec=pmemd_exec,
            cutoff=cutoff,
            temp0=temp, 
            free_energy=True,
            clambda=clambda,
            **defaults['heat'],
            **mask_config
        )

        pres_0_dir = lambda_dir / 'pres_0'
        defaults['pres_0'].update(config.get('pres_0', {}))
        pressurize(
            wdir=pres_0_dir,
            prmtop=prmtop,
            inpcrd=heat_dir / "heat.rst7",
            pmemd_exec=pmemd_exec,
            cutoff=cutoff,
            pressure=pres,
            temp0=temp, 
            free_energy=True, clambda=clambda,
            deffnm='pres_0',
            **defaults['pres_0'],
            **mask_config
        )

        pres_1_dir = lambda_dir / 'pres_1'
        defaults['pres_1'].update(config.get('pres_1', {}))
        pressurize(
            wdir=pres_1_dir,
            prmtop=prmtop,
            inpcrd=pres_0_dir / "pres_0.rst7",
            pmemd_exec=pmemd_exec,
            cutoff=cutoff,
            pressure=pres,
            temp0=temp, 
            free_energy=True, clambda=clambda,
            deffnm='pres_1',
            **defaults['pres_1'],
            **mask_config
        )

        pres_2_dir = lambda_dir / 'pres_2'
        defaults['pres_2'].update(config.get('pres_2', {}))
        pressurize(
            wdir=pres_2_dir,
            prmtop=prmtop,
            inpcrd=pres_1_dir / "pres_1.rst7",
            pmemd_exec=pmemd_exec,
            cutoff=cutoff,
            pressure=pres,
            temp0=temp, 
            free_energy=True, clambda=clambda,
            deffnm='pres_2',
            **defaults['pres_2'],
            **mask_config
        )

        pre_prod_dir = lambda_dir / "pre_prod"
        defaults['pre_prod'].update(config.get('pre_prod', {}))
        pressurize(
            wdir=pre_prod_dir,
            prmtop=prmtop,
            inpcrd=pres_2_dir / "pres_2.rst7",
            pmemd_exec=pmemd_exec,
            cutoff=cutoff,
            pressure=pres,
            temp0=temp, 
            free_energy=True, clambda=clambda,
            deffnm='pre_prod',
            **defaults['pre_prod'],
            **mask_config
        )

        prod_dir = lambda_dir / "prod"
        defaults['prod'].update(config.get('prod', {}))
        prod(
            wdir=prod_dir,
            prmtop=prmtop,
            inpcrd=pres_2_dir / "pres_2.rst7",
            pmemd_exec=pmemd_exec,
            cutoff=cutoff,
            pressure=pres,
            temp0=temp, 
            free_energy=True, clambda=clambda,
            restraint_wt=0.0,
            use_mbar=True,
            deffnm='prod',
            lambdas=lambdas,
            **defaults['prod'],
            **mask_config
        )

    groupfile = []
    str_template = "-O -p {prmtop} -c {inpcrd} -i {mdin} -o {mdout} -r {restart} -x {traj} -ref {ref} -e {mden} -l {mdlog} -inf {mdinfo}"
    for i in range(len(lambdas)):
        groupfile.append(str_template.format(
            prmtop=Path(prmtop).resolve(),
            inpcrd=f"lambda{i}/pre_prod/pre_prod.rst7",
            mdin=f"lambda{i}/prod/prod.in",
            mdout=f"lambda{i}/prod/prod.out",
            restart=f"lambda{i}/prod/prod.rst7",
            traj=f"lambda{i}/prod/prod.mdcrd",
            ref=f"lambda{i}/pre_prod/pre_prod.rst7",
            mden=f"lambda{i}/prod/prod.mden",
            mdlog=f"lambda{i}/prod/prod.log",
            mdinfo=f"lambda{i}/prod/prod.info"
        ))
    with open(wdir / "prod.groupfile", 'w') as f:
        f.write('\n'.join(groupfile))