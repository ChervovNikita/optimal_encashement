# Модель по построение оптимальных маршрутов инкасации платежных терминалов 

Команда **Algoritmistic Heist**

Состав команды: Вольнов Сергей, Червов Никита, Ильин Павел

# Запуск решения
Для того, чтобы запустить данное решение вам необходимо:
1) Установить зависимости из requirements.txt
2) Скачать данные
3) Запустить следующий скрипт
```console
python3 src/main.py --dist_path "data/raw/times v4.csv" --incomes_path "data/raw/terminal_data_hackathon v4.xlsx" --model_path "models/catboost_zero.pkl" --zero_aggregation_path="models/zero_aggregation.pkl" --output_path "data/processed/raw_report.json"
```
Чтобы он отработал данные должны лежать в папке data/raw, а запускаться нужно из корня проекта

Этот скрипт выдаст .json файл с информацией о каждом дне

# Описание работы решения

Наше решение можно разбить на несколько ключевых частей:
1) Предсказание притока наличности в терминалах.
2) Подсчет штрафов для терминалов (как сильно наказывать алгоритм, если мы в этот день не объедем i-ый терминал).
3) Построение оптимальных маршрутов, который объезжают терминалы так, чтобы суммарный штраф по необъеханным терминалам был наименьшим.

**За что платим деньги**
1) сумма процентов в неинкассированных терминалах
2) стоимость обслуживания терминалов
3) стоимость броневиков

Замечание: основную стоимость составляет аренда броневиков, так что их оптимизируем в первую очередь.


## Предсказание притока наличности в терминалах

Посмотрев на данные, мы увидели, что основную проблему для предсказания составляют нулевый притоки наличности. То есть когда терминал был на ремонте или еще почему-то не мог принимать наличность. В остальные дни очень хорошо себя показывает просто брать среднее значение данного терминала.

<img width="997" alt="image" src="https://github.com/ChervovNikita/optimal_encashement/assets/44319901/a5d966ae-1a1a-44d2-ac95-cadff2ea28ee">

В итоге мы построили отдельную модель, которая предсказывает поломки (и соответственно нулевые поступления), а если предсказывается, что все хорошо, то берем среднее значение прироста по данному терминалу.

<img width="720" alt="image" src="https://github.com/ChervovNikita/optimal_encashement/assets/44319901/5aa0c4c4-f2be-4604-8c80-3d1bade3ef11">

**Признаки, использованные для предсказания нулей:**

1) Временные: день недели, число, месяц
2) Праздники и расстояние до них
3) Скользящие средние с разными окнами
4) Лаги (двух дневные, недельные, месячные)
5) Экспонециально взвешенные средние с разными окнами
6) Информация о погоде (спарсили с [сайта](http://weatherarchive.ru/Temperature/Moscow/January-2022) давление, влажность, ветер, температура)
7) Суммарное/среднее кол-во поломок за разные промежутки
8) Время с последней поломки

**Метрики этой модели**

Предсказание нулей: `Accuracy - 0.95, Precision - 0.80, Recall - 0.28`

Предсказание притока: `MAE - 12400, SMAPE - 0.27`

## Определение штрафа на каждый день

Стоимость броневиков составляет основную часть расходов, поэтому наша первостепенная задача это использовать наименьшее кол-во машин, чтобы все терминалы обслуживались в дедлайны, а потом уже оптимизировать то, сколько мы теряем на процентах и самой инкасации.

На идейном уровне штраф определялся так:
* Большой штраф тем, кого надо раньше обслужить
* Если сегодня последний день, то штраф = INF
* Если дедлайн истекает одинаково скоро, то смотрим на delta_loss (экономические потери от того, что сегодня не обслужим, но обслужим завтра)

Формула штрафа за то, что мы сегодня НЕ объедим данный терминал выглядит следующим образом:
<img width="844" alt="image" src="https://github.com/ChervovNikita/optimal_encashement/assets/44319901/46c99928-ac93-4b14-8086-ebe47b241887">


Здесь days_left - это сколько дней осталось до обязательного инкасирования (либо переполнится по нашим прогнозам, либо пройдет две недели с момента последней инкасации)
А delta_loss - экономические потери от того, что сегодня не обслужим, но обслужим завтра (без учета стоимости броневиков).

Примечания: чтобы учитывать delta_loss и soon_deadline_loss с разной силой мы домножаем второй из них на config['inverse_delta_loss']

**Подсчет delta_loss** 
<img width="983" alt="image" src="https://github.com/ChervovNikita/optimal_encashement/assets/44319901/00822a8c-7257-4efa-b304-fdf12e991d51">

Мы знаем, что для любой терминал нам нужно инкасировать хотя бы раз в две недели и как только он переполнился. Получается, если мы знаем, что за k дней нам точно нужно провести один раз инкасаци мы можем определить в какой день это сделать лучше всего. Для этого нужно просто перебрать день инкасации и посчитать потери. Они будут равны тому сколько денег мы потеряем до момента инкасации и со дня инкасации до конца. 

Отлично, мы знаем сколько заплатим, если за i-ый терминал, если будем инкасировать в день d и в день d + 1. Теперь `delta_loss = loss[d + 1] - loss[d]` - солько денег мы потеряем или получим, если вместо того, чтобы инкасировать терминал сегодня сделаем это завтра. И этот штраф добавляется к основному с некоторым коэфицентов.

**Оптимизация при помощи динамического программирования** 

На самом деле, лосс, который считается выше не самый оптимальный, так как он живет в мире, где есть всего одна инкасация. Для того, чтобы смаштабировать это решение на несколько возможных инкасаций мы можем прибегнуть к динамическому программирования. То есть для каждого дня будет считаться наименьший штраф, который можно будет получить в будущем на этом терминале, если мы сегодня его объедем (с учетом того, в какие дни мы объехали его ранее). И delta_loss считается тоже как разница `loss[d + 1]` и `loss[d]`

Простое решение подсчета этих оптимальных лоссов заключается в переборе всех подмножеств дней, в которые мы будем объезжать данные терминал и подсчета штрафа для данной комбинации. Однако такой метод решения работает долго (его ассипмтотика это `O(num_days * 2^num_days)`). Поэтому мы переписали это при помощи динамического программирования и добились ассимптотики `O(days_limit * num_days^2)`, что сильно быстре и куда более масштабируемо.

Примечание: здесь num_days - горизонт планирования (условно 30-90 дней), а days_limit - это как часто нужно обязательно объезжать каждый терминал (в нашем случае это 14 дней).

## Поиск оптимального маршрута

После того, как мы назначили штрафы за то, что сегодня не посетили тот или иной терминал, нам необходимо объехать их всех так, чтобы суммарный штраф по тем терминалам, которые мы не успели объехать был минимален. 

Для этого мы вопспользовались инструментом для маршрутизации от гугла - [ortools](https://developers.google.com/optimization/routing/penalties?hl=ru). 
В нем можно настроить кол-во машин используемых для объезда вершин, ограничение на суммарное кол-во времени в пути для одной машины (длина рабочего дня) и указать штрафы за пропуски. 

<img width="500" alt="image" src="https://github.com/ChervovNikita/optimal_encashement/assets/44319901/5e03f048-d2bd-4418-a358-f9e22009acd2">

## Результаты

Ниже представлены финальные метрики, посчитанные на всем выданном нам промежутке (3 месяца):

1) Суммарная стоимость фондирования = ...
2) Суммарная стоимость инкасации терминалов = ...
3) Кол-во используемых броневиков = 5
4) Стоимость аренды броневиков на все дни = 9.1М
5) Общие расходы = 12.67М
