# 如何執行

## 基本指令

```bash
python3 main.py -p <規劃器> -m <地圖>
```

| 參數 | 選項 |
|------|------|
| `-p` | `a_star` 或 `rrt_star` |
| `-m` | `map1`、`map2`、`map3` |

## 範例

```bash
# A*，地圖1
python3 main.py -p a_star -m map1

# RRT*，地圖2
python3 main.py -p rrt_star -m map2
```

## 注意事項

- 必須在 `~/HW1` 目錄下執行
- 執行後會跳出視覺化視窗，**按任意鍵關閉**
- 若在 VSCode（snap 版）的 terminal 無法執行，請用 `Ctrl+Alt+T` 開系統終端機
