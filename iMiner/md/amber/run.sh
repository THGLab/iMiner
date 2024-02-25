prmtop="{prmtop}"
inpcrd="{inpcrd}"
deffnm="{deffnm}"
{pmemd_exec} -O \
        -i $deffnm.in -o $deffnm.out -p $prmtop -c $inpcrd \
        -r $deffnm.rst7 -inf $deffnm.info -ref $inpcrd \
        -x $deffnm.mdcrd -e $deffnm.mden -l $deffnm.log
ambpdb -p $prmtop -c $deffnm.rst7 > $deffnm.pdb