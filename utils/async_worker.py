# utils/async_worker.py
import threading
import queue
from typing import Callable, Any, Optional
import time
import logging

logger = logging.getLogger(__name__)


class AsyncWorker:
    """
    Gerencia execução assíncrona de tarefas sem travar a GUI.
    """

    def __init__(self):
        self.thread = None
        self.running = False
        self.result_queue = queue.Queue()
        self.error_queue = queue.Queue()

    def run(self, target: Callable, args: tuple = (), kwargs: dict = None,
            callback: Optional[Callable] = None, error_callback: Optional[Callable] = None):
        """
        Executa uma função em thread separada.

        Args:
            target: Função a ser executada
            args: Argumentos posicionais
            kwargs: Argumentos nomeados
            callback: Função chamada quando concluído (recebe resultado)
            error_callback: Função chamada em caso de erro (recebe exceção)
        """
        if self.running:
            logger.warning("Worker já está executando uma tarefa")
            return

        self.running = True
        kwargs = kwargs or {}

        def wrapper():
            try:
                result = target(*args, **kwargs)
                self.result_queue.put(result)
                if callback:
                    # Agendar callback na thread principal
                    self._schedule_callback(callback, result)
            except Exception as e:
                logger.error(f"Erro na thread: {e}")
                self.error_queue.put(e)
                if error_callback:
                    self._schedule_callback(error_callback, e)
            finally:
                self.running = False

        self.thread = threading.Thread(target=wrapper)
        self.thread.daemon = True  # Thread morre quando programa fecha
        self.thread.start()

    def _schedule_callback(self, callback: Callable, arg: Any):
        """
        Agenda callback para execução na thread principal do Tkinter.
        """
        # Esta função será chamada com after() do Tkinter
        # O callback real é agendado no método que chama o worker
        pass

    def is_running(self) -> bool:
        """Verifica se o worker está executando."""
        return self.running and self.thread and self.thread.is_alive()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Aguarda conclusão da tarefa.

        Returns:
            True se concluiu, False se timeout
        """
        if self.thread:
            self.thread.join(timeout)
            return not self.thread.is_alive()
        return True