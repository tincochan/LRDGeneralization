import sys
import azureml.contrib.automl.dnn.forecasting.wrapper.dispatched.invoker.runner as runner

mltable_data_json = None # PLACEHOLDER

if __name__ == '__main__':
    sys.argv.append('--batch_size')
    sys.argv.append('256')
    sys.argv.append('--multilevel')
    sys.argv.append('CELL')
    sys.argv.append('--num_epochs')
    sys.argv.append('100')
    if mltable_data_json:
        runner.run(mltable_data_json=mltable_data_json)
    else:
        runner.run()
