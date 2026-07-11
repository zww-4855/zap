import read_info as read_info


in_file = "/Users/zwu/Desktop/work/bkup/zap/outputs/CH+re/CH+\ Full\ operator/two_elec.txt"
tei = read_info.read_tei(in_file)

print("Two-electron integrals (TEI):",tei)