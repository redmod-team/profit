from profit.run.test import MockupWorker

import asyncio

if __name__ == '__main__':
    worker = MockupWorker.from_env()
    asyncio.run(worker.main())
