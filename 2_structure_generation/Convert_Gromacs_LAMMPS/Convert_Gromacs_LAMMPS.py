import os
import sys
import Fix_Intermol_Gromacs_LAMMPS as fix

def main(gro_file, topol_file, out_folder):
    basename = str.split(gro_file,'/')[-1][:-4]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    Intermol_Convert = os.path.join(script_dir, 'InterMol/intermol/convert.py')
    os.system('echo "In py file"')
    os.system('echo '+Intermol_Convert)
    os.system('python '+Intermol_Convert+' --gro_in '+gro_file+' '+topol_file+
              ' --lammps --odir '+out_folder)
    convert_input = os.path.join(out_folder, basename+'_converted.input')
    convert_lmp = os.path.join(out_folder, basename+'_converted.lmp')
    convert_out = os.path.join(out_folder, basename+'.dat')
    fix.Fix_LAMMPS(convert_input,
                   convert_lmp,
                   convert_out)

if __name__ == "__main__":
    if(len(sys.argv)!=4):
        sys.exit("\n" +
                 " *** IMPORTANT ***\n" +
                 "This script takes two arguments:\n" +
                 "1) Full GROMACS gro input filename plus path\n" +
                 "2) Full GROMACS topology filename plus path\n" +
                 "3) Output folder path\n")
    assert len(sys.argv)==4, "Wrong number of arguments given"
    gro_file = sys.argv[1]
    topol_file = sys.argv[2]
    out_folder = sys.argv[3]
    main(gro_file, topol_file, out_folder)
