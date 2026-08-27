-- 创建数据库
CREATE DATABASE IF NOT EXISTS museumDB;

-- 使用数据库
USE museumDB;

-- 创建博物馆表
CREATE TABLE IF NOT EXISTS museums (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    website VARCHAR(255) NOT NULL
);

-- 插入博物馆数据
INSERT INTO museums (name, website) VALUES
('卢浮宫', 'https://www.louvre.fr/'),
('大都会艺术博物馆', 'https://www.metmuseum.org/'),
('大英博物馆', 'https://www.britishmuseum.org/'),
('故宫博物院', 'https://www.dpm.org.cn/'),
('梵蒂冈博物馆', 'https://www.museivaticani.va/'),
('普拉多博物馆', 'https://www.museodelprado.es/'),
('艾尔米塔什博物馆', 'https://www.hermitagemuseum.org/'),
('奥赛博物馆', 'https://www.musee-orsay.fr/'),
('泰特现代艺术博物馆', 'https://www.tate.org.uk/'),
('国立博物馆', 'https://www.nationalmuseum.org.uk/'),
('纽约现代艺术博物馆', 'https://www.moma.org/'),
('法国国立图书馆', 'https://www.bnf.fr/'),
('东京国立博物馆', 'https://www.tnm.jp/'),
('伦敦自然历史博物馆', 'https://www.nhm.ac.uk/'),
('洛杉矶县艺术博物馆', 'https://www.lacma.org/'),
('柏林博物馆岛', 'https://www.smb.museum/'),
('阿姆斯特丹国立博物馆', 'https://www.rijksmuseum.nl/'),
('巴西国家博物馆', 'https://www.museunacional.ufrj.br/'),
('悉尼现代艺术博物馆', 'https://www.artgallery.nsw.gov.au/');
