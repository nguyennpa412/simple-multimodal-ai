class BaseUtils():
    def start_task(self, task: str) -> None:
        self.curr_tasks.add(task)
        self.num_task = len(self.curr_tasks)

    def done_task(self, task: str) -> None:
        self.curr_tasks.discard(task)
        self.num_task = len(self.curr_tasks)

    def reset_task_tracker(self) -> None:
        self.curr_tasks = set()
        self.num_task = 0