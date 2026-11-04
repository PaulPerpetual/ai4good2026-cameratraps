import sys, pilot_a
pilot_a.TEMPS, pilot_a.GAMMAS = [1.5], [0.5]
sys.argv = ["pilot_a.py"] + sys.argv[1:]
pilot_a.main()
