from Rock_AI.training_jobs.training_process_launcher import TrainingProcessLauncher


def test_launcher_uses_argument_array_without_shell(tmp_path, monkeypatch):
    captured={}
    class Process: pid=9911
    def fake_popen(args, **kwargs): captured["args"]=args; captured["kwargs"]=kwargs; return Process()
    monkeypatch.setattr("Rock_AI.training_jobs.training_process_launcher.subprocess.Popen",fake_popen)
    launcher=TrainingProcessLauncher(); pid=launcher.launch(tmp_path)
    assert pid==9911
    assert isinstance(captured["args"],list)
    assert captured["kwargs"]["shell"] is False
    assert captured["args"][1:4]==["-m","Rock_AI.scripts.run_neat_training_job","--job"]
