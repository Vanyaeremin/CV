import statistics
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    '''
    [LEVEL 2 FIX] pin_memory=True выделяет батчи в "закреплённой" (page-locked) RAM. DMA-контроллер может передавать такую память на GPU напрямую, без промежуточного
    копирования - это ускоряет H2D-трансфер. num_workers>0 загружает и препроцессит следующий батч в отдельных процессах
    параллельно с GPU-вычислениями текущего батча, устраняя bubble простоя GPU.
    '''
    dataloader = DataLoader(
        prepare_data(),
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).cuda().train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses_history = []
    forward_times = []
    backward_times = []

    for batch_idx, (data, target) in enumerate(dataloader):
        '''
        [LEVEL 1 FIX] noise создаём сразу на GPU, а не на CPU с последующим .to('cuda'). 
        Создание на CPU и перенос по PCIe-шине (CPU->GPU) - медленная операция, поэтому указываем device='cuda' сразу.
        '''
        noise = torch.randn(data.shape, device='cuda')

        '''
        [LEVEL 2 FIX] non_blocking=True работает совместно с pin_memory=True.
        Делает H2D-трансфер асинхронным: CPU ставит задачу копирования в CUDA Stream и сразу идёт дальше, не дожидаясь завершения. GPU получает данные по DMA
        параллельно с другой работой.
        '''
        data = data.to('cuda', non_blocking=True) + noise
        target = target.to('cuda', non_blocking=True)
        '''
        [LEVEL 2 FIX] set_to_none=True: вместо заполнения градиентов нулями (лишние write-операции в память GPU) просто освобождаем их,
        что быстрее и экономит память.
        '''
        optimizer.zero_grad(set_to_none=True)
        '''
        [LEVEL 3 FIX] Для честного замера времени GPU-операций используем torch.cuda.Event вместо time.time().
        time.time() на CPU фиксирует лишь момент постановки задачи в CUDA-очередь, а не реальное время её выполнения на GPU.
        '''
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        fwd_start.record()
        output = model(data)
        loss = criterion(output, target)
        fwd_end.record()

        bwd_start.record()
        loss.backward()
        bwd_end.record()

        optimizer.step()
        '''
        [LEVEL 1 FIX] Сохраняем только скалярное значение лосса через .item(), а не сам тензор. 
        Если хранить тензор - он тянет за собой весь вычислительный граф, что приводит к утечке памяти GPU
        и неизбежному OOM на длинных прогонах.
        
        [LEVEL 2 FIX] .item() вызывает синхронизацию CPU<->GPU, поэтому выносим его за пределы "горячего" участка - делаем это после
        optimizer.step(), когда GPU-работа уже поставлена в очередь. Это минимизирует простой GPU в ожидании CPU.
        '''
        loss_value = loss.item()
        losses_history.append(loss_value)
        '''
        [LEVEL 2 FIX] print() с tensor.item() внутри цикла - двойная проблема: синхронизация GPU + медленный IO на каждом батче.
        Логируем только каждые N батчей, чтобы не ломать асинхронный конвейер.
        '''
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} loss: {loss_value:.4f}")
        '''
        [LEVEL 1 FIX] Убираем torch.cuda.empty_cache() из горячего цикла. Эта функция освобождает только кэш аллокатора PyTorch (уже неиспользуемую
        память), но не освобождает живые тензоры. При этом она вызывает синхронизацию и сбивает внутреннее состояние аллокатора,
        что замедляет последующие аллокации. Вызывать её нужно только явно, когда это действительно необходимо (например, между экспериментами).
        '''
        torch.cuda.synchronize()
        forward_times.append(fwd_start.elapsed_time(fwd_end))
        backward_times.append(bwd_start.elapsed_time(bwd_end))

    print(f"Epoch finished, avg forward time is {statistics.mean(forward_times):.3f} ms, "
          f"avg backward time is {statistics.mean(backward_times):.3f} ms")


if __name__ == '__main__':
    train()