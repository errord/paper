# YAML and functional language comparison
_Exported on 2/13/2026 at 12:10:00 GMT+8 from Cursor (2.3.35)_

---

**User**

# ============================================
# 示例：电商-订单限界上下文的 Schema 定义
# ============================================

context: OrderContext
  language:
    Order:       "客户提交的购买请求，包含一个或多个商品项"
    OrderItem:   "订单中的单个商品项，包含商品、数量、单价"
    OrderStatus: "订单在其生命周期中的状态"

  aggregates:
    Order:
      root: Order
      members:
        - OrderItem

      # ---- 聚合根（实体）----
      Order:
        type: entity
        identity: { strategy: uuid }
        properties:
          customer_id:  { type: reference, target: CustomerContext.Customer }
          status:       { type: value_object, ref: OrderStatus }
          items:        { type: collection, of: OrderItem, min: 1 }
          total_amount: { type: money, currency: CNY }
          created_at:   { type: timestamp }
        
        invariants:
          - name: "订单至少包含一个商品项"
            rule: "len(self.items) >= 1"
          - name: "订单总金额等于各项小计之和"
            rule: "self.total_amount == sum(item.subtotal for item in self.items)"
          - name: "已取消的订单不可再次提交"
            rule: "not (self.status == CANCELLED and command == Submit)"

        commands:
          CreateOrder:
            params: [customer_id, items]
            preconditions: ["customer_id 必须存在"]
            postconditions: ["status == CREATED"]
            emits: [OrderCreated]

          SubmitOrder:
            params: []
            preconditions: ["status == CREATED", "items非空"]
            postconditions: ["status == SUBMITTED"]
            emits: [OrderSubmitted]

          CancelOrder:
            params: [reason]
            preconditions: ["status in [CREATED, SUBMITTED]"]
            postconditions: ["status == CANCELLED"]
            emits: [OrderCancelled]

      # ---- 值对象 ----
      OrderItem:
        type: value_object
        properties:
          product_id:   { type: reference, target: ProductContext.Product }
          product_name: { type: string, max_length: 200 }
          quantity:     { type: integer, min: 1 }
          unit_price:   { type: money }
          subtotal:     { type: money, derived: "quantity * unit_price" }

      OrderStatus:
        type: value_object
        enum: [CREATED, SUBMITTED, PAID, SHIPPED, COMPLETED, CANCELLED]
        transitions:
          CREATED:   [SUBMITTED, CANCELLED]
          SUBMITTED: [PAID, CANCELLED]
          PAID:      [SHIPPED]
          SHIPPED:   [COMPLETED]

    # ---- 领域事件 ----
    events:
      OrderCreated:
        payload: { order_id, customer_id, items, total_amount }
        subscribers: [NotificationContext, AnalyticsContext]

      OrderSubmitted:
        payload: { order_id, customer_id, total_amount }
        subscribers: [InventoryContext, PaymentContext]

      OrderCancelled:
        payload: { order_id, reason }
        subscribers: [InventoryContext, NotificationContext]

    # ---- 上下文映射 ----
    mappings:
      - target: InventoryContext
        type: partnership            # 合作关系
        protocol: domain_events      # 通过领域事件通信
      
      - target: PaymentContext
        type: customer_supplier      # 客户-供应商关系
        protocol: api_call           # 同步API调用
        anti_corruption_layer: true  # 需要防腐层

      - target: CustomerContext
        type: conformist             # 遵从者
        protocol: shared_kernel      # 共享内核

上面的yaml转为lisp或scheme语言等纯函数式语言来表达

这样一个YAML的配置文件和一个纯函数式语言文件这2类形式化描述相比，从人类可读及机器可读以及AI可读等方面全面比较。最后哪一种是最好的？

---

**Cursor**

## YAML vs 纯函数式语言（S-expression）形式化描述对比

### 一、先将 YAML 转为 Scheme S-expression 表达

```scheme
;; ============================================
;; 电商-订单限界上下文 Schema 定义 (Scheme S-expression)
;; ============================================

(define-context OrderContext

  ;; ---- 统一语言 ----
  (language
    (Order       "客户提交的购买请求，包含一个或多个商品项")
    (OrderItem   "订单中的单个商品项，包含商品、数量、单价")
    (OrderStatus "订单在其生命周期中的状态"))

  ;; ---- 聚合 ----
  (aggregate Order
    (root   Order)
    (members OrderItem)

    ;; ---- 聚合根（实体）----
    (entity Order
      (identity uuid)

      (properties
        (customer_id  (reference CustomerContext.Customer))
        (status       (value-object OrderStatus))
        (items        (collection OrderItem (min 1)))
        (total_amount (money CNY))
        (created_at   timestamp))

      (invariants
        ("订单至少包含一个商品项"
         (>= (length (self-ref items)) 1))

        ("订单总金额等于各项小计之和"
         (= (self-ref total_amount)
            (sum (map (lambda (item) (item-ref item subtotal))
                      (self-ref items)))))

        ("已取消的订单不可再次提交"
         (not (and (= (self-ref status) 'CANCELLED)
                   (= command 'Submit)))))

      (commands
        (CreateOrder
          (params customer_id items)
          (preconditions  ("customer_id 必须存在"))
          (postconditions ((= status 'CREATED)))
          (emits OrderCreated))

        (SubmitOrder
          (params)
          (preconditions  ((= status 'CREATED) "items非空"))
          (postconditions ((= status 'SUBMITTED)))
          (emits OrderSubmitted))

        (CancelOrder
          (params reason)
          (preconditions  ((member status '(CREATED SUBMITTED))))
          (postconditions ((= status 'CANCELLED)))
          (emits OrderCancelled))))

    ;; ---- 值对象 ----
    (value-object OrderItem
      (properties
        (product_id   (reference ProductContext.Product))
        (product_name (string (max-length 200)))
        (quantity     (integer (min 1)))
        (unit_price   money)
        (subtotal     (money (derived (* quantity unit_price))))))

    (value-object OrderStatus
      (enum CREATED SUBMITTED PAID SHIPPED COMPLETED CANCELLED)
      (transitions
        (CREATED   -> SUBMITTED CANCELLED)
        (SUBMITTED -> PAID CANCELLED)
        (PAID      -> SHIPPED)
        (SHIPPED   -> COMPLETED))))

  ;; ---- 领域事件 ----
  (events
    (OrderCreated
      (payload order_id customer_id items total_amount)
      (subscribers NotificationContext AnalyticsContext))

    (OrderSubmitted
      (payload order_id customer_id total_amount)
      (subscribers InventoryContext PaymentContext))

    (OrderCancelled
      (payload order_id reason)
      (subscribers InventoryContext NotificationContext)))

  ;; ---- 上下文映射 ----
  (mappings
    (mapping InventoryContext
      (type partnership)
      (protocol domain-events))

    (mapping PaymentContext
      (type customer-supplier)
      (protocol api-call)
      (anti-corruption-layer #t))

    (mapping CustomerContext
      (type conformist)
      (protocol shared-kernel))))
```

---

### 二、全面对比分析

#### 1. 人类可读性（Human Readability）

| 维度 | YAML | S-expression (Scheme) |
|------|------|----------------------|
| **视觉结构** | ⭐⭐⭐⭐⭐ 缩进即结构，几乎是"所见即所得" | ⭐⭐⭐ 括号嵌套深时需要编辑器辅助 |
| **学习门槛** | ⭐⭐⭐⭐⭐ 非技术人员也能快速理解 | ⭐⭐ 需要理解S-expression和括号匹配 |
| **领域专家友好度** | ⭐⭐⭐⭐⭐ 业务人员可直接审阅 | ⭐⭐ 业务人员会被括号吓退 |
| **信噪比** | ⭐⭐⭐⭐ 少量语法噪音（冒号、短横线） | ⭐⭐⭐ 括号本身是噪音，但关键字更精确 |
| **规则表达力** | ⭐⭐⭐ 不变量用字符串嵌入，本质是伪代码 | ⭐⭐⭐⭐⭐ 不变量就是真实的可执行表达式 |

**小结**：YAML 在"第一眼可读性"上完胜。但 YAML 的不变量（invariants）和派生规则（derived）本质上是**字符串中的伪代码**，读起来简单但语义模糊。S-expression 的规则是**真正的表达式**，对懂 Lisp 的人来说反而更精确。

---

#### 2. 机器可读性（Machine Readability）

| 维度 | YAML | S-expression (Scheme) |
|------|------|----------------------|
| **解析复杂度** | ⭐⭐⭐ 需要YAML解析器，规范复杂（锚点、引用、多行字符串等各种坑） | ⭐⭐⭐⭐⭐ 解析器极其简单（递归下降，~50行代码） |
| **类型安全** | ⭐⭐ 一切皆字符串，类型靠约定 | ⭐⭐⭐⭐ 天然有 symbol、list、string、number 区分 |
| **规则可执行性** | ⭐ 不变量是字符串，需二次解析 | ⭐⭐⭐⭐⭐ 不变量就是合法表达式，可直接 `eval` |
| **代码生成** | ⭐⭐⭐ 需要遍历 dict/map 结构 | ⭐⭐⭐⭐⭐ AST 天然就是 list，宏系统可直接变换 |
| **歧义性** | ⭐⭐ YAML 有著名的 "Norway problem"（`NO` 被解析为 `false`） | ⭐⭐⭐⭐⭐ 几乎零歧义，语法完全确定 |
| **可组合/可扩展** | ⭐⭐ 扩展靠约定标签 | ⭐⭐⭐⭐⭐ 宏系统天然支持 DSL 扩展 |

**小结**：S-expression 在机器处理方面**碾压式优势**。YAML 最大的问题是：
- **规则是死字符串**：`"self.total_amount == sum(item.subtotal for item in self.items)"` 对机器来说只是一段文本，需要另外写解析器。
- **类型歧义**：YAML 的隐式类型转换是出了名的坑（`on` → `true`，`3.10` → `3.1`）。
- 而 S-expression 的不变量 `(= (self-ref total_amount) (sum (map ...)))` **就是 AST**，可以直接求值、变换、生成代码。

---

#### 3. AI 可读性（AI/LLM Readability）

| 维度 | YAML | S-expression (Scheme) |
|------|------|----------------------|
| **训练数据丰富度** | ⭐⭐⭐⭐⭐ 海量 YAML 在训练集中 | ⭐⭐⭐ Lisp/Scheme 数据量较少 |
| **结构识别准确度** | ⭐⭐⭐⭐ 缩进结构容易被LLM理解 | ⭐⭐⭐⭐ 括号结构明确无歧义 |
| **语义精确度** | ⭐⭐⭐ 规则是字符串，LLM需"猜"语义 | ⭐⭐⭐⭐⭐ 规则是结构化表达式，语义完全确定 |
| **生成准确度** | ⭐⭐⭐ 缩进容易生成出错（多一个空格就变义） | ⭐⭐⭐⭐ 括号匹配虽难但语义不会因空格变化 |
| **推理友好度** | ⭐⭐⭐ 嵌入字符串的规则需要LLM自行解析 | ⭐⭐⭐⭐⭐ 规则即代码，可直接参与推理链 |
| **上下文窗口效率** | ⭐⭐⭐⭐ 较紧凑 | ⭐⭐⭐ 括号略占空间，但信息密度相当 |

**小结**：对 AI 来说，这是一个**精度 vs 熟悉度**的权衡：
- YAML 赢在 LLM 见过太多 YAML，"直觉"好。
- S-expression 赢在**语义无歧义**——AI 做代码生成、规则验证、形式化推理时，S-expression 提供的结构化信息远胜于 YAML 中嵌入的伪代码字符串。

---

### 三、核心矛盾与最终结论

两种格式的根本区别在于一个关键问题：

> **规则/不变量/派生逻辑 —— 是"数据"还是"代码"？**

| | YAML | S-expression |
|--|------|-------------|
| 属性定义 | ✅ 擅长（纯数据描述） | ✅ 同样胜任 |
| 结构关系 | ✅ 擅长（层级嵌套） | ✅ 同样胜任 |
| **业务规则** | ❌ **字符串伪代码，死的** | ✅ **可执行表达式，活的** |
| **状态转换** | ❌ 只是枚举，无法验证 | ✅ 可直接变成状态机 |
| **代码生成** | 🔶 需要额外解析层 | ✅ AST 即数据，直接变换 |

### 🏆 最佳选择：**混合方案 / S-expression 胜出**

**如果只能选一个，S-expression 是更优选择**，理由如下：

1. **DDD Schema 不是纯配置，它包含行为规则**。YAML 适合描述"是什么"（属性、枚举），但 DDD 的核心价值在于"业务规则的形式化"——不变量、前置/后置条件、状态转换。这些本质是**逻辑表达式**，S-expression 天然就是。

2. **"代码即数据"（Homoiconicity）是杀手锏**。S-expression 写出的 Schema **本身就是 AST**，可以：
   - 直接 `eval` 验证不变量
   - 用宏展开生成实体类代码
   - 做形式化验证（证明状态转换的完备性）
   - AI 可以直接在结构上做推理，而不需要先"理解"字符串中的伪代码

3. **YAML 的"易读"是假象**。当业务规则写成 `"self.total_amount == sum(item.subtotal for item in self.items)"` 这种字符串时，它**既不是合法的 YAML 语义，也不是合法的 Python**——它是两头不靠的伪代码。而 S-expression 版本的 `(= (self-ref total_amount) (sum (map ...)))` 是**真正可执行的**。

4. **如果在意人类可读性，推荐混合方案**：

```scheme
;; 最佳实践：S-expression 内核 + 文档注释
(entity Order
  ;; 📋 客户提交的购买请求
  (identity uuid)
  (properties
    (customer_id  (reference CustomerContext.Customer))  ; 下单客户
    (status       (value-object OrderStatus))            ; 当前状态
    (items        (collection OrderItem (min 1)))        ; 商品项列表
    (total_amount (money CNY)))                          ; 订单总额

  ;; 🔒 业务不变量（可直接执行验证）
  (invariant "订单至少包含一个商品项"
    (>= (length (self-ref items)) 1))

  (invariant "总金额 = Σ各项小计"
    (= (self-ref total_amount)
       (sum (map subtotal (self-ref items))))))
```

这样既保留了 S-expression 的**机器精确性和可执行性**，又通过注释提供了**人类可读的业务语言**。

**一句话总结**：
> YAML 是给人看的配置格式，S-expression 是给机器和人共用的知识表示格式。DDD Schema 本质是**知识表示**而非配置，所以 S-expression（或类 Lisp DSL）是更正确的选择。

```plaintext
;; ============================================
;; 电商-订单限界上下文 Schema 定义 (Scheme S-expression)
;; ============================================

(define-context OrderContext

  ;; ---- 统一语言 ----
  (language
    (Order       "客户提交的购买请求，包含一个或多个商品项")
    (OrderItem   "订单中的单个商品项，包含商品、数量、单价")
    (OrderStatus "订单在其生命周期中的状态"))

  ;; ---- 聚合 ----
  (aggregate Order
    (root   Order)
    (members OrderItem)

    ;; ---- 聚合根（实体）----
    (entity Order
      (identity uuid)

      (properties
        (customer_id  (reference CustomerContext.Customer))
        (status       (value-object OrderStatus))
        (items        (collection OrderItem (min 1)))
        (total_amount (money CNY))
        (created_at   timestamp))

      (invariants
        ("订单至少包含一个商品项"
         (>= (length (self-ref items)) 1))

        ("订单总金额等于各项小计之和"
         (= (self-ref total_amount)
            (sum (map (lambda (item) (item-ref item subtotal))
                      (self-ref items)))))

        ("已取消的订单不可再次提交"
         (not (and (= (self-ref status) 'CANCELLED)
                   (= command 'Submit)))))

      (commands
        (CreateOrder
          (params customer_id items)
          (preconditions  ("customer_id 必须存在"))
          (postconditions ((= status 'CREATED)))
          (emits OrderCreated))

        (SubmitOrder
          (params)
          (preconditions  ((= status 'CREATED) "items非空"))
          (postconditions ((= status 'SUBMITTED)))
          (emits OrderSubmitted))

        (CancelOrder
          (params reason)
          (preconditions  ((member status '(CREATED SUBMITTED))))
          (postconditions ((= status 'CANCELLED)))
          (emits OrderCancelled))))

    ;; ---- 值对象 ----
    (value-object OrderItem
      (properties
        (product_id   (reference ProductContext.Product))
        (product_name (string (max-length 200)))
        (quantity     (integer (min 1)))
        (unit_price   money)
        (subtotal     (money (derived (* quantity unit_price))))))

    (value-object OrderStatus
      (enum CREATED SUBMITTED PAID SHIPPED COMPLETED CANCELLED)
      (transitions
        (CREATED   -> SUBMITTED CANCELLED)
        (SUBMITTED -> PAID CANCELLED)
        (PAID      -> SHIPPED)
        (SHIPPED   -> COMPLETED))))

  ;; ---- 领域事件 ----
  (events
    (OrderCreated
      (payload order_id customer_id items total_amount)
      (subscribers NotificationContext AnalyticsContext))

    (OrderSubmitted
      (payload order_id customer_id total_amount)
      (subscribers InventoryContext PaymentContext))

    (OrderCancelled
      (payload order_id reason)
      (subscribers InventoryContext NotificationContext)))

  ;; ---- 上下文映射 ----
  (mappings
    (mapping InventoryContext
      (type partnership)
      (protocol domain-events))

    (mapping PaymentContext
      (type customer-supplier)
      (protocol api-call)
      (anti-corruption-layer #t))

    (mapping CustomerContext
      (type conformist)
      (protocol shared-kernel))))
```

```plaintext
;; 最佳实践：S-expression 内核 + 文档注释
(entity Order
  ;; 📋 客户提交的购买请求
  (identity uuid)
  (properties
    (customer_id  (reference CustomerContext.Customer))  ; 下单客户
    (status       (value-object OrderStatus))            ; 当前状态
    (items        (collection OrderItem (min 1)))        ; 商品项列表
    (total_amount (money CNY)))                          ; 订单总额

  ;; 🔒 业务不变量（可直接执行验证）
  (invariant "订单至少包含一个商品项"
    (>= (length (self-ref items)) 1))

  (invariant "总金额 = Σ各项小计"
    (= (self-ref total_amount)
       (sum (map subtotal (self-ref items))))))
```

