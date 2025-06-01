from collections import deque


class SparseGradBuffer:
    def __init__(self, m: int = 10) -> None:
        self.m = m
        self.dq = deque()
        self.cnt = 0

    def push(self, x: object) -> None:
        if self.cnt == self.m:
            self.pop()
        self.dq.appendleft(x)
        self.cnt += 1

    def pop(self) -> None:
        self.dq.pop()
        self.cnt -= 1

    def __len__(self) -> int:
        return self.m

    def __getitem__(self, idx: int) -> object:
        return self.dq[idx]
