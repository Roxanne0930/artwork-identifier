// backend/server.js
const express = require('express');
const cors = require('cors');
const pool = require('./db');

const app = express();
const PORT = 3000;

// 启用CORS，允许跨域请求
app.use(cors());

// 搜索API - 路径为/api/search
app.get('/api/search', async (req, res) => {
    try {
        const query = req.query.name;
        if (!query) {
            return res.status(400).json({ error: '请提供搜索关键词' });
        }

        // 执行数据库查询
        const [rows] = await pool.execute(
            'SELECT * FROM museums WHERE name LIKE ?',
            [`%${query}%`]
        );

        if (rows.length === 0) {
            return res.status(404).json({ error: '未找到匹配的博物馆' });
        }

        // 返回查询结果
        res.json(rows);
    } catch (error) {
        console.error('数据库查询错误:', error);
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// 启动服务器
app.listen(PORT, () => {
    console.log(`服务器运行在 http://localhost:${PORT}`);
});