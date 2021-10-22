import itertools
import subprocess

hyperparams = {
        #"--batch_size": [48, 64, 72],
        "--batch-size": [48, 64],
        #"-T": [3, 4, 5, 6],
        "-T": [3, 4, 5, 6],
        #"-M": [4, 5, 6],
        "-M": [5],
        #"--lr": [0.001, 0.01, 0.1]
        }
keys, values = zip(*hyperparams.items())
param_choices = itertools.product(*values)

runNo = 40  # Update this
for param_choices in itertools.product(*values):
    opts = []
    for k,v in zip(keys, param_choices):
        opts += [k, str(v)]

    jobname = "mpnn03-repeat{:03d}".format(runNo)
    core_command = ["python", 
                    "deepchem_model.py", jobname]
    command = core_command + opts
    print(command)
    try:
        completed = subprocess.run(command,
                stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as err:
        print('Error: ', err)
    else:
        print('Return code: ', completed.returncode)
        print("{}".format(completed.stdout.decode('utf-8')))

    runNo += 1

