use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;

use persia_libs::anyhow::{anyhow, Result};
use persia_libs::tracing;

use persia_common::HashMapEmbeddingEntry;
use persia_speedy::{Readable, Writable};

const INIT_BUFFER_SIZE: usize = 1000;

#[derive(Readable, Writable, Debug)]
pub struct PerisaIncrementalPacket {
    pub content: Vec<(u64, Vec<f32>)>,
    pub timestamps: u64,
}

#[derive(Readable, Writable, Debug)]
pub enum SpeedyObj {
    EmbeddingVec(Vec<(u64, HashMapEmbeddingEntry)>),
    PerisaIncrementalPacket(PerisaIncrementalPacket),
}

pub trait PersiaStorageVisitor: Send + Sync {
    fn create_file(&self, file_dir: PathBuf, file_name: PathBuf) -> Result<PathBuf>;

    fn read_from_file(&self, file_path: PathBuf) -> Result<Vec<u8>>;

    fn read_from_file_speedy(&self, file_path: PathBuf) -> Result<SpeedyObj>;

    fn dump_to_file(&self, content: Vec<u8>, file_dir: PathBuf, file_name: PathBuf) -> Result<()>;

    fn dump_to_file_speedy(
        &self,
        content: SpeedyObj,
        file_dir: PathBuf,
        file_name: PathBuf,
    ) -> Result<()>;

    fn is_file(&self, file_path: PathBuf) -> Result<bool>;

    fn list_dir(&self, dir_path: PathBuf) -> Result<Vec<PathBuf>>;

    fn remove_file(&self, file_path: PathBuf) -> Result<()>;

    fn append_line_to_file(&self, line: String, file_path: PathBuf) -> Result<()>;
}

struct PersiaDiskVisitor {}

impl PersiaStorageVisitor for PersiaDiskVisitor {
    fn create_file(&self, file_dir: PathBuf, file_name: PathBuf) -> Result<PathBuf> {
        std::fs::create_dir_all(file_dir.clone())?;
        let file_path: PathBuf = [file_dir, file_name].iter().collect();
        if file_path.is_file() {
            return Err(anyhow!("file already exist"));
        }
        let _out_file = File::create(file_path.clone())?;

        Ok(file_path)
    }

    fn read_from_file(&self, file_path: PathBuf) -> Result<Vec<u8>> {
        let mut f = File::open(&file_path)?;
        let metadata = std::fs::metadata(&file_path)?;
        let mut buffer = vec![0; metadata.len() as usize];
        f.read(&mut buffer)?;

        Ok(buffer)
    }

    fn read_from_file_speedy(&self, file_path: PathBuf) -> Result<SpeedyObj> {
        let content: SpeedyObj = SpeedyObj::read_from_file(file_path)?;
        Ok(content)
    }

    fn dump_to_file(&self, content: Vec<u8>, file_dir: PathBuf, file_name: PathBuf) -> Result<()> {
        let file_path = self.create_file(file_dir.clone(), file_name.clone())?;

        let out_file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(file_path.to_str().unwrap())?;
        tracing::debug!("success to open file");

        let mut buffered = BufWriter::new(out_file);
        buffered.write_all(content.as_slice())?;
        buffered.flush()?;

        Ok(())
    }

    fn dump_to_file_speedy(
        &self,
        content: SpeedyObj,
        file_dir: PathBuf,
        file_name: PathBuf,
    ) -> Result<()> {
        let file_path = self.create_file(file_dir, file_name)?;
        content.write_to_file(&file_path)?;
        Ok(())
    }

    fn is_file(&self, file_path: PathBuf) -> Result<bool> {
        Ok(file_path.is_file())
    }

    fn list_dir(&self, dir_path: PathBuf) -> Result<Vec<PathBuf>> {
        let mut res = Vec::new();
        let paths = dir_path.read_dir()?;
        for p in paths {
            let p = p?.path();
            res.push(p);
        }
        Ok(res)
    }

    fn remove_file(&self, file_path: PathBuf) -> Result<()> {
        std::fs::remove_file(file_path)?;
        Ok(())
    }

    fn append_line_to_file(&self, line: String, file_path: PathBuf) -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open(file_path.to_str().unwrap())?;
        writeln!(file, "{}", line)?;
        Ok(())
    }
}

struct PersiaHdfsVisitor {}

impl PersiaStorageVisitor for PersiaHdfsVisitor {
    fn create_file(&self, file_dir: PathBuf, file_name: PathBuf) -> Result<PathBuf> {
        let mkdir_out = Command::new("hdfs")
            .arg("dfs")
            .arg("-mkdir")
            .arg("-p")
            .arg(file_dir.as_os_str())
            .output()?;
        if !mkdir_out.status.success() {
            return Err(anyhow!("hdfs mkdir error"));
        }

        let file_path: PathBuf = [file_dir, file_name].iter().collect();
        if self.is_file(file_path.clone())? {
            return Err(anyhow!("file already exist"));
        }

        let touch_out = Command::new("hdfs")
            .arg("dfs")
            .arg("-touchz")
            .arg(file_path.as_os_str())
            .output()?;

        if touch_out.status.success() {
            Ok(file_path)
        } else {
            Err(anyhow!("hdfs touchz error"))
        }
    }

    fn read_from_file(&self, file_path: PathBuf) -> Result<Vec<u8>> {
        let text_cmd = Command::new("hadoop")
            .arg("fs")
            .arg("-text")
            .arg(file_path.as_os_str())
            .stdout(Stdio::piped())
            .spawn()?;

        let mut stdout = text_cmd.stdout.unwrap();
        let mut result = Vec::with_capacity(INIT_BUFFER_SIZE);
        stdout.read_to_end(&mut result)?;
        Ok(result)
    }

    fn read_from_file_speedy(&self, file_path: PathBuf) -> Result<SpeedyObj> {
        let text_cmd = Command::new("hadoop")
            .arg("fs")
            .arg("-text")
            .arg(file_path.as_os_str())
            .stdout(Stdio::piped())
            .spawn()?;

        let stdout = text_cmd.stdout.unwrap();
        let content: SpeedyObj = SpeedyObj::read_from_stream_buffered(BufReader::new(stdout))?;

        Ok(content)
    }

    fn dump_to_file(&self, content: Vec<u8>, file_dir: PathBuf, file_name: PathBuf) -> Result<()> {
        let file_path = self.create_file(file_dir.clone(), file_name)?;
        let mut append_cmd = Command::new("hdfs")
            .arg("dfs")
            .arg("-appendToFile")
            .arg("-")
            .arg(file_path.as_os_str())
            .stdin(Stdio::piped())
            .spawn()?;

        let write_stream = BufWriter::new(append_cmd.stdin.as_mut().unwrap());
        content.write_to_stream(write_stream)?;

        drop(append_cmd.stdin.as_mut().unwrap());

        let out = append_cmd.wait()?;
        if out.success() {
            return Ok(());
        } else {
            return Err(anyhow!("hdfs appendToFile error"));
        }
    }

    fn dump_to_file_speedy(
        &self,
        content: SpeedyObj,
        file_dir: PathBuf,
        file_name: PathBuf,
    ) -> Result<()> {
        let file_path = self.create_file(file_dir.clone(), file_name)?;
        let mut append_cmd = Command::new("hdfs")
            .arg("dfs")
            .arg("-appendToFile")
            .arg("-")
            .arg(file_path.as_os_str())
            .stdin(Stdio::piped())
            .spawn()?;

        let write_stream = BufWriter::new(append_cmd.stdin.as_mut().unwrap());
        content.write_to_stream(write_stream)?;

        drop(append_cmd.stdin.as_mut().unwrap());

        let out = append_cmd.wait()?;
        if out.success() {
            return Ok(());
        } else {
            return Err(anyhow!("hdfs appendToFile error"));
        }
    }

    fn is_file(&self, file_path: PathBuf) -> Result<bool> {
        let test_out = Command::new("hdfs")
            .arg("dfs")
            .arg("-test")
            .arg("-e")
            .arg(file_path.as_os_str())
            .output()?;

        let res = test_out.status.success();
        Ok(res)
    }

    fn list_dir(&self, dir_path: PathBuf) -> Result<Vec<PathBuf>> {
        let ls_out = Command::new("hdfs")
            .arg("dfs")
            .arg("-ls")
            .arg(dir_path.as_os_str())
            .output()?
            .stdout;

        let s = std::str::from_utf8(&ls_out);
        if s.is_err() {
            return Err(anyhow!("hdfs ls error"));
        }
        let mut file_list = Vec::new();
        for f in s.unwrap().split_whitespace() {
            if f.starts_with("hdfs://") {
                file_list.push(PathBuf::from(f));
            }
        }
        Ok(file_list)
    }

    fn remove_file(&self, file_path: PathBuf) -> Result<()> {
        let rm_out = Command::new("hdfs")
            .arg("dfs")
            .arg("-rm")
            .arg(file_path.as_os_str())
            .output()?;
        if rm_out.status.success() {
            Ok(())
        } else {
            Err(anyhow!("hdfs rm error"))
        }
    }

    fn append_line_to_file(&self, line: String, file_path: PathBuf) -> Result<()> {
        let mut file_dir = file_path.clone();
        file_dir.pop();

        let file_name = file_path.file_name();
        if file_name.is_none() {
            return Err(anyhow!("can not parse file_name error"));
        }
        let file_name = PathBuf::from(file_name.unwrap());

        self.create_file(file_dir, file_name)?;

        let mut append_cmd = Command::new("hdfs")
            .arg("dfs")
            .arg("-appendToFile")
            .arg("-")
            .arg(file_path.as_os_str())
            .stdin(Stdio::piped())
            .spawn()?;

        let stdin = append_cmd.stdin.as_mut().unwrap();
        stdin.write_all(line.as_bytes())?;

        drop(stdin);
        let append_out = append_cmd.wait_with_output()?;
        if append_out.status.success() {
            Ok(())
        } else {
            Err(anyhow!("hdfs appendToFile error"))
        }
    }
}

pub struct PersiaStorageAdapter {
    disk_visitor: Arc<PersiaDiskVisitor>,
    hdfs_visitor: Arc<PersiaHdfsVisitor>,
}

impl PersiaStorageAdapter {
    pub fn new() -> Self {
        Self {
            disk_visitor: Arc::new(PersiaDiskVisitor {}),
            hdfs_visitor: Arc::new(PersiaHdfsVisitor {}),
        }
    }

    fn get_visitor(&self, path: &PathBuf) -> Arc<dyn PersiaStorageVisitor> {
        match path.starts_with("hdfs://") {
            true => self.hdfs_visitor.clone(),
            false => self.disk_visitor.clone(),
        }
    }

    pub fn create_file(&self, file_dir: PathBuf, file_name: PathBuf) -> Result<PathBuf> {
        self.get_visitor(&file_dir).create_file(file_dir, file_name)
    }

    pub fn read_from_file(&self, file_path: PathBuf) -> Result<Vec<u8>> {
        self.get_visitor(&file_path).read_from_file(file_path)
    }

    pub fn read_from_file_speedy(&self, file_path: PathBuf) -> Result<SpeedyObj> {
        self.get_visitor(&file_path)
            .read_from_file_speedy(file_path)
    }

    pub fn dump_to_file(
        &self,
        content: Vec<u8>,
        file_dir: PathBuf,
        file_name: PathBuf,
    ) -> Result<()> {
        self.get_visitor(&file_dir)
            .dump_to_file(content, file_dir, file_name)
    }

    pub fn dump_to_file_speedy(
        &self,
        content: SpeedyObj,
        file_dir: PathBuf,
        file_name: PathBuf,
    ) -> Result<()> {
        self.get_visitor(&file_dir)
            .dump_to_file_speedy(content, file_dir, file_name)
    }

    pub fn is_file(&self, file_path: PathBuf) -> Result<bool> {
        self.get_visitor(&file_path).is_file(file_path)
    }

    pub fn list_dir(&self, dir_path: PathBuf) -> Result<Vec<PathBuf>> {
        self.get_visitor(&dir_path).list_dir(dir_path)
    }

    pub fn remove_file(&self, file_path: PathBuf) -> Result<()> {
        self.get_visitor(&file_path).remove_file(file_path)
    }

    pub fn append_line_to_file(&self, line: String, file_path: PathBuf) -> Result<()> {
        self.get_visitor(&file_path)
            .append_line_to_file(line, file_path)
    }
}
