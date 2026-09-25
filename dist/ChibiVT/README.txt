МОДЕЛЬ ДЛЯ VTube Studio — ChibiVT
=====================================

ЧТО ВНУТРИ
  ChibiVT.model3.json     главный файл модели (его ищет VTube Studio)
  ChibiVT.moc3            геометрия, деформеры, параметры
  ChibiVT.physics3.json   физика: корпус догоняет поворот головы
  ChibiVT.cdi3.json       служебная информация (не обязательна)
  textures/texture_00.png  текстура персонажа (2048x1024)
  icon.png               иконка модели в списке VTube Studio

КАК УСТАНОВИТЬ
  1. Скопируйте ВСЮ папку ChibiVT целиком (не отдельные файлы!) в папку
     Live2DModels вашего VTube Studio. Открыть её можно кнопкой
     «Open Data Folder»/«Open Folder» в настройках VTube Studio
     (Steam-версия), либо вручную:
       Windows : %%USERPROFILE%%\Documents\VTube Studio\Live2DModels
                 (или Data\Live2DModels рядом с приложением)
       macOS   : ~/Library/Application Support/VTube Studio/Data/Live2DModels
  2. Запустите VTube Studio — модель появится в списке моделей.
  3. При первом открытии согласитесь на «Auto-Setup»: VTube Studio сам
     привяжет трекинг лица к стандартным параметрам модели.

ПАРАМЕТРЫ (стандартные Live2D — трекинг работает сразу)
  ParamAngleX / Y / Z        поворот, наклон и крен головы   (-30..30)
  ParamBodyAngleX / Y / Z    корпус                          (-10..10)
  ParamEyeLOpen / ParamEyeROpen  моргание (1 — открыт, 0 — закрыт)
  ParamMouthOpenY            открытие рта (лип-синк)
  ParamMouthForm             не задействован (параметр можно добавить)

ЕСЛИ ЧТО-ТО НЕ РАБОТАЕТ
  • Модель не видна — проверьте, что папка скопирована целиком и лежит
    именно в Live2DModels (а не в подпапке подпапки).
  • Голова/рот двигаются слабо — в настройках модели VTube Studio
    (шестерёнка → Model Movement / VTS Parameter Setup) увеличьте OUT-значения
    у FaceAngleX/Y/Z и MouthOpen.
  • Корпус не двигается — включите Body Angle в настройках модели или
    увеличьте вес в ChibiVT.physics3.json (поле Weight).
  • Не нравится физика — удалите ChibiVT.physics3.json и ссылку на него
    в ChibiVT.model3.json: модель продолжит работать без физики.

ПЕРЕСБОРКА (если нужно подправить риг)
  Модель собирается из одного изображения скриптами в каталоге tools/:
      python3 tools/rig.py               # текстура: глаза/рот вырезаны, атлас
      python3 tools/build_model.py       # меши, деформеры, параметры -> .moc3
      python3 tools/validate.py          # проверка через Cubism Core
      python3 tools/poses.py             # контактный лист поз для проверки
  Полный цикл пересборки — эти четыре команды по порядку.

ЧТО ГДЕ КРУТИТЬ
  tools/build_model.py, словарь CFG (в начале файла):
      yaw_scale / pitch_scale / roll_scale — амплитуда поворота головы
      r_yaw / r_pitch                     — «толщина» головы (перспектива)
      eye_closed_scale                    — насколько сжимается глаз при моргании
      body_shift_x / body_shift_y         — наклон корпуса
  tools/rig.py, cfg в конце файла (__main__):
      mouth_open_w / mouth_open_h         — размер открытого рта
      eye_dark_thresh / mouth_dark_thresh — порог, по которому ищутся глаза и рот
