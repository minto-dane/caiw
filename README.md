# caiw — Compressor for AI Weights

**Lossless, byte-exact archiver for safetensors model weights.**
One C file. No dependencies beyond libc + pthread. Multithreaded.

Generic compressors see model weights as opaque bytes. caiw understands
the bit structure of `F16`/`BF16`/`F32` tensors — sign, exponent and
mantissa fields — plus cross-tensor structure: duplicates, checkpoint
deltas, external reference weights. Every tensor is compressed with
whichever method *actually produces the fewest bytes*, and every archive
reconstructs its inputs bit-for-bit (CRC32-verified).

*日本語での詳しい使い方は本 README 末尾の「[使い方（日本語）](#使い方日本語)」をご覧ください。*

## Results

Qwen3-4B shard 3 (BF16, 99.6 MB), `-j8`, this machine:

| tool | size | ratio | wall time |
|---|---|---|---|
| **caiw** | **65.78 MB** | **0.660** | **3.2 s** |
| xz -9e | 70.15 MB | 0.704 | 85.9 s |
| zstd -19 -T8 | 76.52 MB | 0.768 | 26.4 s |
| gzip -9 | 79.22 MB | 0.795 | 7.8 s |

Full Qwen3-4B (8.0 GB, 3 shards, 398 tensors): **5.32 GB (0.662)**,
encoded in 4m40s, verified `0 bad`.

MoE shard (4.0 GB, 1063 tensors): **2.65 GB (0.663)**, ~2 min encode.
OLMo-2 F32 pair (5.0 GB) against its base snapshot via `--ref`:
**2.79 GB (0.560)** — 170 of 173 tensors resolved as `DELTAX`.

## Quick start

```console
$ make                      # cc -O3 -march=native, pthread + libm only
$ ./caiw c model.caiw model.st -j8        # compress
$ ./caiw v model.caiw model.st -j8        # verify byte-exact
$ ./caiw d model.caiw outdir/ -j8         # decompress to safetensors
```

Checkpoint deltas (e.g. store a fine-tune against its base):

```console
$ ./caiw c tune.caiw --ref base.st tune.st -j8
$ ./caiw d tune.caiw outdir/ --ref base.st   # same --ref set, same order
```

`make test` generates deterministic fixtures and runs the full
regression (round-trips at -j1/-j4, metadata, and a checked-in
crash corpus that must die cleanly).

## How it works

**Competition, not heuristics.** Every candidate encoder runs against a
clone of the same entropy-model state; the smallest *real* output wins.
Cheap integer estimates may *prune* a hopeless candidate before it runs —
they never pick the winner — and if nothing helps, RAW wins.

| method | exploits |
|---|---|
| `FIELD` | sign/exponent/mantissa of f16-family, context-conditioned rANS |
| `FIELDT` | FIELD with the model transmitted as a frozen table — blocks decode in parallel (`CAI6`) |
| `FIELDPOS`/`FIELDROW` | + column/row-position context (embeddings, rotary tables) |
| `F32` | 5-plane field decomposition for f32 |
| `DELTA` | residual vs an earlier same-name/family tensor (checkpoint chains) |
| `DELTAX` | residual vs an external `--ref` checkpoint |
| `PRW` | residual vs the previous row of the same tensor (smooth rows) |
| `REF` | content-hash exact dedup |
| `PACK` | restricted alphabets (≤256 distinct atoms → bit-packed indices) |
| `U8` | per-byte entropy fallback |
| `RAW` | store — the honest floor |

**Parallelism without nondeterminism.** Candidate trials and consecutive
tensors encode in parallel (batches): each worker sees a snapshot
of the committed histogram state, histogram deltas merge commutatively,
and the archive records batch boundaries so an archive written at *any*
`-j` decodes to identical bytes under *any* `-j`. Every stream encoder
is additionally *block-parallel* internally. On the decode side, FIELD's
adaptive table chains blocks serially — so FIELDT normalizes the model
over the whole tensor once, transmits that table in the payload (dense
sign/exponent rows; a mantissa row only where it provably beats uniform
coding), and each block then decodes independently across threads. On
real-weight tensors FIELDT beats FIELD outright on bytes (~-1.3% on
pythia-160m) *and* decodes ~3.8x faster at `-j8`; archives containing it
are stamped `CAI6`, anything else stays `CAI5` for older readers.

**Rolling context models.** Each channel keeps a cumulative histogram;
per-block tables are normalized only over contexts the block actually
uses (used-mask), which the decoder reconstructs identically from the
already-decoded reference tensor.

## Correctness and robustness

- **Byte-exact by construction** — `v` re-derives every tensor and
  compares bytes + CRC32 in both directions (archive vs source, incl.
  dtype/shape/`__metadata__`); `d` restores every tensor and the
  verbatim metadata object.
- **Hostile-input safe** — every malformed archive or truncated
  safetensors file exits(1) through `die()`; the decoder validates every
  field, method/dtype pair, payload bound, and batch reference before
  touching memory.
- **Verified** — ASan/UBSan/MSan/TSan clean; a checked-in crash
  corpus (`tests/corpus/`) replays inputs that once crashed. CI
  (`ci.yml`) rebuilds from a clean Ubuntu image and reruns `test.sh`
  under -O3/-Werror/ASan+UBSan and on real aarch64, plus every
  `verify.sh` layer — TLA+/Alloy/CBMC in the `verify` job, Frama-C
  EVA+WP + Rocq proofs in `verify-framac` (pinned opam switch +
  pinned z3). Formal layer under `verify/`
  (`verify/verify.sh`): TLA+ model-checks the batch-merge and
  tmp→rename protocols (flawed variants produce the expected
  counterexamples), Alloy checks archive structural invariants at
  scope 8, CBMC proves kmap16/dsym/norm_ctx/rANS (fixed-frequency)
  plus the parser boundary (utf8_ok/hex4/jstr/jstr_skip/jspan),
  Frama-C EVA+RTE screens the codec kernels (0 invalid), and Frama-C
  WP discharges the ACSL contracts — 3022/3022 goals across 65
  functions covering every payload decoder (U8/FIELD/FIELDPOS/
  FIELDROW/F32/PACK/DELTA/DELTAX/PRW): memory safety, cursor
  bounds, and exact payload consumption on untrusted input, an
  unbounded proof per contract. Five Rocq/Coq proofs (zero-axiom,
  coqchk-rechecked) cover the regions every BMC backend timed out on:
  rANS round trip over the full general domain, utf8_ok correctness
  at any length, norm_ctx's output contract over all histograms,
  PACK bitstream round trip at any length, and jkey's nested scan
  (bounds, termination, key-position correctness). Bounded results
  and solver limits are documented honestly; the rANS core is
  additionally covered by an exhaustive boundary sweep (589,809
  cases — `verify/sweeps.c`, run by `verify.sh`) and `dsym` by
  3,264 tables × all 32,768 symbol values.
- **Deterministic** — encode decisions are integer-only, so a fixed `-j`
  reproduces bit-identical archives across machines; archives are
  `-j`-independent on the decode side.

See `AGENTS.md` for the format specification, the decoder trust-boundary
checklist, and the full verification log.

---

## 使い方（日本語）

caiw は safetensors 形式のモデルウェイトを**バイト完全一致**で圧縮・復元する
アーカイバです。単一の C ファイルのみで構成され、依存は libc・libm・pthread
だけ。汎用コンプレッサがモデルウェイトを不透明なバイト列として扱うのに対し、
caiw は `F16`/`BF16`/`F32` テンソルのビット構造（符号・指数・仮数）と、
チェックポイント間の関係（重複・差分・外部参照）を理解して圧縮します。
すべてのテンソルは「実際に最もバイト数が少なかったメソッド」で圧縮され、
すべてのアーカイブは入力をビット単位で再現します（CRC32 検証済み）。

### 実測結果

Qwen3-4B シャード3（BF16、99.6MB）、`-j8`、本機での計測：

| ツール | サイズ | 圧縮率 | 時間 |
|---|---|---|---|
| **caiw** | **65.78 MB** | **0.660** | **3.2 s** |
| xz -9e | 70.15 MB | 0.704 | 85.9 s |
| zstd -19 -T8 | 76.52 MB | 0.768 | 26.4 s |
| gzip -9 | 79.22 MB | 0.795 | 7.8 s |

その他の実測：

- **Qwen3-4B 全体**（8.0GB、3シャード、398テンソル）：**5.32GB (0.662)**、
  エンコード 4分40秒、`0 bad` で検証完了
- **MoE シャード**（4.0GB、1063テンソル）：**2.65GB (0.663)**、約2分
- **OLMo-2 F32 ペア**（5.0GB、`--ref` でベーススナップショット参照）：
  **2.79GB (0.560)** — 173 テンソル中 170 が `DELTAX` に解決

### ビルド

```console
$ make          # cc -O3 -march=native, pthread + libm のみ
```

直接ビルドする場合：

```console
$ gcc -O3 -march=native -Wall -Wextra -pthread -o caiw caiw.c -lm
```

### 基本的な操作

| コマンド | 動作 |
|---|---|
| `c` | safetensors を圧縮し、アーカイブを作成します |
| `v` | アーカイブの内容を元ファイルと**双方向でバイト照合**します |
| `d` | アーカイブを safetensors として復元します |

```console
$ ./caiw c model.caiw model.st -j8        # 圧縮
$ ./caiw v model.caiw model.st -j8        # 検証（0 bad で完全一致）
$ ./caiw d model.caiw outdir/ -j8         # outdir/ 以下に復元
```

複数の safetensors をひとつのアーカイブにまとめられます：

```console
$ ./caiw c bundle.caiw shard-01.st shard-02.st shard-03.st -j8
```

### スレッド数 `-j N`

`-j8` / `-j 8` のどちらでも指定できます。省略時はオンラインコア数
（上限 8）、許容範囲は 1〜64 で、`-j1` は完全な逐次実行です。

- エンコード時：候補メソッドの試行と連続テンソルのバッチ処理、さらに各
  エンコーダ内部のブロック単位処理が並列に実行されます
- デコード時：**どの `-j` で作成されたアーカイブも、どの `-j` でも同一の
  バイト列に復元**されます（FIELDT テンソルはブロック単位で並列復号）
- 同一 `-j` では機種を問わず bit-identical なアーカイブが得られます。
  `-j` が異なるとバッチ境界が変わるため、アーカイブのバイト列自体が
  変わることがあります（復元結果は常に同一です）

### チェックポイント差分 `--ref`

ファインチューニング済みモデルなどを、ベースモデルに対する差分として
保存できます：

```console
$ ./caiw c tuned.caiw --ref base.st tuned.st -j8
$ ./caiw d tuned.caiw outdir/ --ref base.st   # 同じ --ref を同じ順序で
```

`--ref` は繰り返し指定でき、テンソルごとに最も効率の良い参照が自動選択
されます（`DELTAX`）。アーカイブに記録されるのは参照ファイルの
**インデックス**です。復元時はそのインデックスを優先し、次に名前・
サイズ・dtype が一致するテンソルを全 `--ref` から探索します。

> **注意**：`c` と `d` で `--ref` の集合と順序を一致させてください。
> 互換性のある別の参照が選ばれて誤ったバイト列になった場合も、
> CRC32 チェックが必ず検出して失敗します——静かに壊れた出力が
> 残ることはありません。

### 検証 `v` の意味

`v` は双方向の完全一致検証です：

- アーカイブ内の全テンソルが、元ファイル内に同名・同 dtype・同 shape・
  同バイト列で存在すること
- 元ファイル側の全テンソルがアーカイブに含まれること（MISS-SRC）
- 重複テンソル名がないこと（DUP）、`__metadata__` が一致すること（META）

`0 bad` と表示されれば完全に一致しています。`d` なしに整合性だけを
確認したい場合に最適です。

### 出力の安全性

- 出力は一時ファイルに書き込まれ、全テンソルの成功後に `rename` で
  本配置されます（all-or-nothing）。失敗時は一時ファイルが自動除去されます
- 全テンソルに CRC32 が付き、破損は必ず `die()` で報告されます。
  CRC32 は改ざん防止ではなく**破損検出**です。来歴が重要な場合は
  アーカイブ全体に別途ハッシュ・署名を付けてください

### 仕組み

**ヒューリスティックではなく競争です。** すべての候補エンコーダは、同じ
エントロピーモデル状態のクローンに対して実際に符号化を行い、
**出力が最も小さかったものだけが採用**されます。安価な整数推定は
見込みのない候補を事前に*枝刈り*するだけで、勝者を選ぶことは
ありません。何も効かなければ `RAW`（無圧縮）が正直に勝ちます。

| メソッド | 狙い |
|---|---|
| `FIELD` | f16 系の符号・指数・仮数をコンテキスト条件付き rANS で符号化 |
| `FIELDT` | FIELD のモデルを凍結テーブルとして転送——ブロックが独立し並列復号（`CAI6`） |
| `FIELDPOS`/`FIELDROW` | 列・行位置をコンテキストに追加（埋め込み・回転テーブル向け） |
| `F32` | f32 の 5 プレーンフィールド分解 |
| `DELTA` | 同一アーカイブ内の先行テンソルとの差分（チェックポイント連鎖） |
| `DELTAX` | 外部 `--ref` チェックポイントとの差分 |
| `PRW` | 同一テンソルの前の行との差分（滑らかな行を持つ行列向け） |
| `REF` | 内容ハッシュによる完全一致重複排除 |
| `PACK` | 限定アルファベット（≤256 種の値をビットパック） |
| `U8` | バイト単位エントロピー符号化（フォールバック） |
| `RAW` | 無圧縮格納——正直な下限 |

**非決定性を持ち込まない並列化。** 候補の試行と連続テンソルは並列に
エンコードされます（バッチ処理）。各ワーカーはコミット済みヒストグラム
状態のスナップショットを見て動き、ヒストグラムの差分は可換に
マージされ、アーカイブにはバッチ境界が記録されます。そのため
**どの `-j` で書かれたアーカイブも、どの `-j` でも同一バイト列に
復元**されます。各ストリームエンコーダは内部でもブロック並列です。
デコード側では FIELD の適応テーブルがブロックを逐次連鎖させるため、
FIELDT はテンソル全体でモデルを一度だけ正規化し、そのテーブルを
ペイロードに転送します（符号・指数行は密に、仮数行は一様符号化を
確実に上回る場合のみ）。各ブロックは独立に、スレッドを跨いで復号
できます。実際のウェイト分布では FIELDT は FIELD よりバイト数でも
勝り（pythia-160m で約 -1.3%）、`-j8` では復号も約 3.8 倍高速です。
FIELDT を含むアーカイブは `CAI6`、それ以外は `CAI5` のままです。

**ローリングコンテキストモデル。** 各チャネルは累積ヒストグラムを保持し、
ブロック毎のテーブルはそのブロックが実際に使うコンテキストだけで
正規化されます（used-mask）。デコーダ側は復号済み参照テンソルから
同一の集合を再構成します。

### フォーマット互換性

| magic | 内容 |
|---|---|
| `CAI6` | FIELDT テンソルを含むアーカイブ（並列復号可能） |
| `CAI5` | 現在の標準形式（`__metadata__` を保持） |
| `CAI4`/`CAI3` | 過去バージョン——このバージョンでも復号可能 |

FIELDT を含まないアーカイブは `CAI5` のままなので、古いデコーダでも
読み続けられます。

### 検証体制

正しさは複数の独立した層で担保されています：

- **往復検証**——`v` が全テンソルのバイト列・dtype・shape・
  `__metadata__` を双方向で照合し、CRC32 で破損を検出します
- **敵対入力への耐性**——不正アーカイブや切り詰められた safetensors は
  必ず `die()` で `exit(1)`。フィールド・メソッド/dtype 組合せ・
  ペイロード境界・バッチ参照をメモリに触れる前にすべて検証します
- **サニタイザ**——ASan/UBSan/MSan/TSan すべてクリーン。
  `tests/corpus/` には過去にクラッシュを引き起こした入力の
  回帰コーパスが残っています
- **形式検証**（`verify/`、`verify/verify.sh`）——TLA+ がバッチマージと
  tmp→rename プロトコルをモデル検査（欠陥版は期待通り反例を生成）、
  Alloy がアーカイブ構造不変条件をスコープ 8 で検査、CBMC が
  kmap16/dsym/norm_ctx/rANS とパーサ境界を証明、Frama-C EVA+RTE が
  コーデックカーネルを走査、Frama-C WP が ACSL 契約を検証
  （65 関数・3022 ゴール、全ペイロードデコーダをカバー）、
  Rocq/Coq がゼロ公理証明 5 本（一般領域 rANS 往復、utf8_ok、
  norm_ctx 出力契約、PACK 往復、jkey ネスト走査）を提供します。
  有界な結果とソルバーの限界は正直に文書化してあり、rANS コアは
  さらに網羅的境界スイープ（589,809 ケース）でも確認済みです
- **決定性**——エンコード判断は整数演算のみ。同一 `-j` なら機種を
  問わず bit-identical なアーカイブが再現されます

### 制限事項

- 要素ペイロードはホストのバイトオーダで解釈されます——アーカイブの
  相互運用は**リトルエンディアン環境のみ**（x86 / ARM-LE）。
  ビッグエンディアンでは CRC が失敗して止まり、静かな破損は起きません
- `d` は出力ファイルをすべて同時に開きます。アーカイブ内のファイル数が
  `RLIMIT_NOFILE` を超える場合は `ulimit` を上げてください
- mmap 中の入力を別プロセスが同時に切り詰めた場合の SIGBUS は
  OS レベルの競合であり、ユーザ空間では完全には防げません
- メンバー名の一意性はバイト完全一致です——大文字小文字を区別しない
  ファイルシステム（APFS/NTFS）では別名が衝突し得ます

テストは `make test`（または `./test.sh`）で実行できます。決定論的な
フィクスチャ生成、`-j1`/`-j4` の往復、メタデータ往復、クラッシュコーパスの
拒否まで含む完全なリグレッションが走ります。

フォーマット仕様・デコーダの信頼境界チェック一覧・検証ログの全文は
`AGENTS.md` に、アルゴリズムの詳しい解説（論文形式）は
`docs/design.md` にあります。
