# 增强版AI股票分析系统 v3.1
# 本版本还未完善请勿使用。

## 系统概述

增强版AI股票分析系统是一个基于Python的Web应用，支持对A股、港股和美股进行全面分析。系统集成了技术分析、基本面分析和情绪分析，并利用AI模型生成专业的投资建议。

主要特点：
- 支持中国A股、香港股票和美国股票市场
- AI流式输出，实时生成分析结果
- 多维度分析：技术指标、财务数据、市场情绪
- 自定义提示词和分析参数
- 新闻原文提供给AI，增强分析深度
- 历史交易数据可视化

## 配置文件详解

系统通过`config.json`文件进行配置，以下是完整的配置项说明：

### API密钥配置

```json
"api_keys": {
    "openai": "sk-your-openai-api-key-here",
    "anthropic": "sk-ant-your-claude-api-key-here",
    "zhipu": "your-zhipu-api-key-here",
    "notes": "请填入您的API密钥，至少配置一个"
}
```

| 配置项 | 说明 |
|-------|------|
| openai | OpenAI API密钥，用于GPT模型 |
| anthropic | Anthropic API密钥，用于Claude模型 |
| zhipu | 智谱AI API密钥，用于ChatGLM模型 |

### AI模型配置

```json
"ai": {
    "model_preference": "openai",
    "models": {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "zhipu": "chatglm_turbo"
    },
    "max_tokens": 4000,
    "temperature": 0.7,
    "api_base_urls": {
        "openai": "https://api.openai.com/v1",
        "notes": "如使用中转API，修改上述URL"
    }
}
```

| 配置项 | 说明 |
|-------|------|
| model_preference | 首选AI服务提供商，可选值：openai、anthropic、zhipu |
| models | 各提供商使用的模型名称 |
| max_tokens | 生成文本的最大token数量 |
| temperature | 生成文本的创造性程度，值越高创造性越强 |
| api_base_urls | API基础URL，可配置中转API |

### 分析权重配置

```json
"analysis_weights": {
    "technical": 0.4,
    "fundamental": 0.4,
    "sentiment": 0.2,
    "notes": "权重总和应为1.0"
}
```

| 配置项 | 说明 |
|-------|------|
| technical | 技术分析权重，影响综合评分 |
| fundamental | 基本面分析权重，影响综合评分 |
| sentiment | 情绪分析权重，影响综合评分 |

### 缓存配置

```json
"cache": {
    "price_hours": 1,
    "fundamental_hours": 6,
    "news_hours": 2
}
```

| 配置项 | 说明 |
|-------|------|
| price_hours | 价格数据缓存时间（小时） |
| fundamental_hours | 基本面数据缓存时间（小时） |
| news_hours | 新闻数据缓存时间（小时） |

### 流式输出配置

```json
"streaming": {
    "enabled": true,
    "show_thinking": false,
    "delay": 0.05
}
```

| 配置项 | 说明 |
|-------|------|
| enabled | 是否启用AI流式输出 |
| show_thinking | 是否显示AI思考过程 |
| delay | 流式输出延迟（秒） |

### 分析参数配置

```json
"analysis_params": {
    "max_news_count": 100,
    "technical_period_days": 180,
    "financial_indicators_count": 25,
    "include_news_content": true,
    "max_news_tokens": 2000,
    "recent_trading_days": 30,
    "hide_scores": true
}
```

| 配置项 | 说明 |
|-------|------|
| max_news_count | 最大新闻分析数量 |
| technical_period_days | 技术分析周期（天） |
| financial_indicators_count | 财务指标数量 |
| include_news_content | 是否在AI分析中包含新闻原文 |
| max_news_tokens | 新闻内容的最大token数量 |
| recent_trading_days | 提供给AI的历史交易天数 |
| hide_scores | 是否隐藏评分（不提供给AI） |

### 自定义提示词配置

```json
"custom_prompts": {
    "enabled": true,
    "analysis_template": "",
    "notes": "自定义提示词模板，留空则使用系统默认模板"
}
```

| 配置项 | 说明 |
|-------|------|
| enabled | 是否启用自定义提示词 |
| analysis_template | 自定义提示词模板，支持变量替换 |

### 第三方数据源配置

```json
"external_sources": {
    "enabled": false,
    "sources": [
        {
            "name": "第三方数据源示例",
            "url": "https://example.com/api/data",
            "api_key": "",
            "enabled": false
        }
    ],
    "notes": "第三方数据源配置，用于辅助预测"
}
```

| 配置项 | 说明 |
|-------|------|
| enabled | 是否启用第三方数据源 |
| sources | 数据源列表，包含名称、URL和API密钥 |

### 市场配置

```json
"markets": {
    "a_stock": {
        "enabled": true,
        "currency": "CNY",
        "timezone": "Asia/Shanghai",
        "trading_hours": "09:30-15:00",
        "notes": "中国A股市场"
    },
    "hk_stock": {
        "enabled": true,
        "currency": "HKD", 
        "timezone": "Asia/Hong_Kong",
        "trading_hours": "09:30-16:00",
        "notes": "香港股票市场"
    },
    "us_stock": {
        "enabled": true,
        "currency": "USD",
        "timezone": "America/New_York", 
        "trading_hours": "09:30-16:00",
        "notes": "美国股票市场"
    }
}
```

| 配置项 | 说明 |
|-------|------|
| enabled | 是否启用该市场 |
| currency | 市场货币 |
| timezone | 市场时区 |
| trading_hours | 交易时间 |

### Web认证配置

```json
"web_auth": {
    "enabled": false,
    "password": "",
    "session_timeout": 3600,
    "notes": "Web界面密码鉴权配置，可选启用"
}
```

| 配置项 | 说明 |
|-------|------|
| enabled | 是否启用Web认证 |
| password | 访问密码 |
| session_timeout | 会话超时时间（秒） |

### 元数据

```json
"_metadata": {
    "version": "3.1.0-multi-market-streaming",
    "description": "增强版AI股票分析系统配置文件（支持A股/港股/美股 + AI流式输出）",
    "last_updated": "2024-01-15"
}
```

| 配置项 | 说明 |
|-------|------|
| version | 配置文件版本 |
| description | 配置文件描述 |
| last_updated | 最后更新日期 |

## 自定义提示词模板变量

在自定义提示词模板中，可以使用以下变量：

### 基本信息变量
- `{{stock_code}}` - 股票代码
- `{{stock_name}}` - 股票名称
- `{{current_price}}` - 当前价格
- `{{price_change}}` - 价格变化百分比
- `{{volume_ratio}}` - 成交量比率
- `{{volatility}}` - 波动率

### 市场信息变量
- `{{market_info}}` - 完整的市场信息块
- `{{market_name}}` - 市场名称（如"中国A股市场"）

### 技术分析变量
- `{{ma_trend}}` - 均线趋势
- `{{rsi}}` - RSI指标值
- `{{macd_signal}}` - MACD信号
- `{{bb_position}}` - 布林带位置
- `{{volume_status}}` - 成交量状态

### 情绪分析变量
- `{{overall_sentiment}}` - 整体情绪得分
- `{{sentiment_trend}}` - 情绪趋势
- `{{confidence_score}}` - 置信度
- `{{total_analyzed}}` - 分析的新闻总数

### 评分变量（当hide_scores=false时可用）
- `{{technical_score}}` - 技术面得分
- `{{fundamental_score}}` - 基本面得分
- `{{sentiment_score}}` - 情绪面得分
- `{{comprehensive_score}}` - 综合得分

### 其他数据块
- `{{financial_text}}` - 财务指标详情
- `{{news_content}}` - 新闻内容
- `{{external_data}}` - 第三方数据源信息
- `{{recent_trading_data}}` - 最近交易数据

## 启动系统

1. 确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

2. 配置`config.json`文件（可从`config -示例.json`复制并修改）

3. 启动Web服务器：
```bash
python enhanced_flask_server.py
```

4. 在浏览器中访问：`http://localhost:5000`

## 注意事项

1. 至少需要配置一个AI API密钥才能使用AI分析功能
2. 数据获取可能受到网络和API限制，建议适当设置缓存时间
3. 分析结果仅供参考，不构成投资建议
4. 使用流式输出可能会增加API调用成本

## 版本历史

- v3.1.0: 支持多市场（A股/港股/美股）+ AI流式输出 + 自定义提示词
- v2.6.0: 添加流式传输测试版
- v2.5.0: 添加Web应用支持
- v2.0.0: 添加Windows应用支持
- v1.0.0: 初始版本，基本分析功能
