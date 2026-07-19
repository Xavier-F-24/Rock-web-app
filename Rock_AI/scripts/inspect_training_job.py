"""Print safe status and progress for one local training job."""

import argparse, json
from Rock_AI.training_jobs.training_progress_reader import TrainingProgressReader

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--job", required=True); args=parser.parse_args(argv)
    reader=TrainingProgressReader(args.job); print(json.dumps({"status": reader.status().to_dict(), "progress": reader.progress()}, indent=2))
if __name__ == "__main__": main()
