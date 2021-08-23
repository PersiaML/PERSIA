use anyhow::{anyhow, Result};
use persia_speedy::{Readable, Writable};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use persia_embedding_datatypes::HashMapEmbeddingEntry;

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

pub struct PersiaCephVisitor {}

impl PersiaStorageVisitor for PersiaCephVisitor {
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

        let out_file = File::open(file_path)?;

        let mut buffered = BufWriter::new(out_file);
        buffered.write_all(content.as_ref())?;
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

pub struct PersiaHdfsVisitor {
    local_buffer_dir: PathBuf,
    local_storage_visitor: PersiaCephVisitor,
}

impl PersiaHdfsVisitor {
    pub fn new() -> Self {
        let local_buffer_dir = PathBuf::from("/tmp/persia_buffer_dir/");
        std::fs::create_dir_all(local_buffer_dir.clone())
            .expect("perisa can not create dump buffer dir");
        Self {
            local_buffer_dir,
            local_storage_visitor: PersiaCephVisitor {},
        }
    }
}

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
        let get_out = Command::new("hdfs")
            .arg("dfs")
            .arg("-get")
            .arg(file_path.as_os_str())
            .arg(self.local_buffer_dir.as_os_str())
            .output()?;

        if get_out.status.success() {
            let local_buffer_file: PathBuf = [
                self.local_buffer_dir.clone().as_os_str(),
                file_path.file_name().unwrap(),
            ]
            .iter()
            .collect();
            let res = self
                .local_storage_visitor
                .read_from_file(local_buffer_file.clone());
            self.local_storage_visitor.remove_file(file_path.clone())?;

            res
        } else {
            Err(anyhow!("hdfs get error"))
        }
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
        let local_buffer_file: PathBuf = [self.local_buffer_dir.clone(), file_name.clone()]
            .iter()
            .collect();

        self.local_storage_visitor.dump_to_file(
            content,
            self.local_buffer_dir.clone(),
            file_name.clone(),
        )?;

        let mkdir_out = Command::new("hdfs")
            .arg("dfs")
            .arg("-mkdir")
            .arg("-p")
            .arg(file_dir.as_os_str())
            .output()?;

        if !mkdir_out.status.success() {
            return Err(anyhow!("hdfs mkdir error"));
        }

        let put_out = Command::new("hdfs")
            .arg("dfs")
            .arg("-put")
            .arg(local_buffer_file.as_os_str())
            .arg(file_dir.as_os_str())
            .output()?;

        if !put_out.status.success() {
            return Err(anyhow!("hdfs put error"));
        }

        self.local_storage_visitor.remove_file(local_buffer_file)
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
            if f.starts_with(dir_path.to_str().unwrap()) {
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
