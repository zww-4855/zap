awk '{print $2}' qsceom_energy | head -25 | awk '{ print ($1 + 37.86730835322125) * 27.2114 }'

cat CASCI_output.txt | head -25 | awk '{ print ($1 + 37.867464458898) * 27.2114 }'
