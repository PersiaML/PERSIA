use enum_dispatch::enum_dispatch;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use persia_libs::anyhow::{anyhow, Result};
use persia_speedy::{LittleEndian, Readable, Writable};

const INIT_BUFFER_SIZE: usize = 1000;

#[enum_dispatch]
pub enum PersiaPath {
    Disk(PersiaDiskPathImpl),
    Hdfs(PersiaHdfsPathImpl),
}

impl PersiaPath {
    pub fn from_pathbuf(path: PathBuf) -> Self {
        match path.starts_with("hdfs://") {
            true => PersiaHdfsPathImpl { inner: path }.into(),
            false => PersiaDiskPathImpl { inner: path }.into(),
        }
    }

    pub fn from_string(s: String) -> Self {
        let path = PathBuf::from(s);
        Self::from_pathbuf(path)
    }

    pub fn from_str(s: &str) -> Self {
        let path = PathBuf::from(s);
        Self::from_pathbuf(path)
    }

    pub fn from_vec(v: Vec<&PathBuf>) -> Self {
        let path: PathBuf = v.iter().collect();
        Self::from_pathbuf(path)
    }
}

#[enum_dispatch(PersiaPath)]
pub trait PersiaPathImpl {
    fn create(&self, p: bool) -> Result<()>;

    fn parent(&self) -> Result<PathBuf>;

    fn is_file(&self) -> Result<bool>;

    fn read_to_end(&self) -> Result<Vec<u8>>;

    fn read_to_end_speedy<'a, R>(&self) -> Result<R>
    where
        R: Readable<'a, LittleEndian>;

    fn write_all(&self, content: Vec<u8>) -> Result<()>;

    fn write_all_speedy<W>(&self, content: W) -> Result<()>
    where
        W: Writable<LittleEndian>;

    fn list(&self) -> Result<Vec<PathBuf>>;

    fn remove(&self) -> Result<()>;

    fn append(&self, line: String) -> Result<()>;
}

pub struct PersiaDiskPathImpl {
    inner: PathBuf,
}

impl PersiaPathImpl for PersiaDiskPathImpl {
    fn parent(&self) -> Result<PathBuf> {
        let path = self
            .inner
            .parent()
            .ok_or_else(|| anyhow!("parent not exist"))?;
        Ok(PathBuf::from(path))
    }

    fn create(&self, p: bool) -> Result<()> {
        let parent = self.parent()?;
        std::fs::create_dir_all(parent)?;
        if self.is_file()? && !p {
            return Err(anyhow!("file already exist"));
        }
        let _out_file = File::create(self.inner.clone())?;

        Ok(())
    }

    fn is_file(&self) -> Result<bool> {
        Ok(self.inner.is_file())
    }

    fn read_to_end(&self) -> Result<Vec<u8>> {
        let mut f = File::open(&self.inner)?;
        let metadata = std::fs::metadata(&self.inner)?;
        let mut buffer = vec![0; metadata.len() as usize];
        f.read(&mut buffer)?;

        Ok(buffer)
    }

    fn read_to_end_speedy<'a, R>(&self) -> Result<R>
    where
        R: Readable<'a, LittleEndian>,
    {
        let content = R::read_from_file(self.inner.clone())?;
        Ok(content)
    }

    fn write_all(&self, content: Vec<u8>) -> Result<()> {
        self.create(false)?;
        let out_file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .open(self.inner.to_str().unwrap())?;

        let mut buffered = BufWriter::new(out_file);
        buffered.write_all(content.as_slice())?;
        buffered.flush()?;

        Ok(())
    }

    fn write_all_speedy<W>(&self, content: W) -> Result<()>
    where
        W: Writable<LittleEndian>,
    {
        self.create(false)?;
        content.write_to_file(&self.inner)?;
        Ok(())
    }

    fn list(&self) -> Result<Vec<PathBuf>> {
        let mut res = Vec::new();
        let paths = self.inner.read_dir()?;
        for p in paths {
            let p = p?.path();
            res.push(p);
        }
        Ok(res)
    }

    fn remove(&self) -> Result<()> {
        std::fs::remove_file(self.inner.clone())?;
        Ok(())
    }

    fn append(&self, line: String) -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open(self.inner.to_str().unwrap())?;
        writeln!(file, "{}", line)?;
        Ok(())
    }
}

pub struct PersiaHdfsPathImpl {
    inner: PathBuf,
}

impl PersiaPathImpl for PersiaHdfsPathImpl {
    fn create(&self, p: bool) -> Result<()> {
        let parent = self.parent()?;
        let mkdir_out = Command::new("hdfs")
            .arg("dfs")
            .arg("-mkdir")
            .arg("-p")
            .arg(parent.as_os_str())
            .output()?;
        if !mkdir_out.status.success() {
            return Err(anyhow!("hdfs mkdir error"));
        }

        if self.is_file()? && !p {
            return Err(anyhow!("file already exist"));
        }

        let touch_out = Command::new("hdfs")
            .arg("dfs")
            .arg("-touchz")
            .arg(self.inner.as_os_str())
            .output()?;

        if touch_out.status.success() {
            Ok(())
        } else {
            Err(anyhow!("hdfs touchz error"))
        }
    }

    fn parent(&self) -> Result<PathBuf> {
        let path = self
            .inner
            .parent()
            .ok_or_else(|| anyhow!("parent not exist"))?;
        Ok(PathBuf::from(path))
    }

    fn is_file(&self) -> Result<bool> {
        let test_out = Command::new("hdfs")
            .arg("dfs")
            .arg("-test")
            .arg("-e")
            .arg(self.inner.as_os_str())
            .output()?;

        let res = test_out.status.success();
        Ok(res)
    }

    fn read_to_end(&self) -> Result<Vec<u8>> {
        let text_cmd = Command::new("hadoop")
            .arg("fs")
            .arg("-text")
            .arg(self.inner.as_os_str())
            .stdout(Stdio::piped())
            .spawn()?;

        let mut stdout = text_cmd.stdout.unwrap();
        let mut result = Vec::with_capacity(INIT_BUFFER_SIZE);
        stdout.read_to_end(&mut result)?;
        Ok(result)
    }

    fn read_to_end_speedy<'a, R>(&self) -> Result<R>
    where
        R: Readable<'a, LittleEndian>,
    {
        let text_cmd = Command::new("hadoop")
            .arg("fs")
            .arg("-text")
            .arg(self.inner.as_os_str())
            .stdout(Stdio::piped())
            .spawn()?;

        let stdout = text_cmd.stdout.unwrap();
        let content: R = R::read_from_stream_buffered(BufReader::new(stdout))?;

        Ok(content)
    }

    fn write_all(&self, content: Vec<u8>) -> Result<()> {
        self.create(false)?;

        let mut append_cmd = Command::new("hdfs")
            .arg("dfs")
            .arg("-appendToFile")
            .arg("-")
            .arg(self.inner.as_os_str())
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

    fn write_all_speedy<W>(&self, content: W) -> Result<()>
    where
        W: Writable<LittleEndian>,
    {
        self.create(false)?;
        let mut append_cmd = Command::new("hdfs")
            .arg("dfs")
            .arg("-appendToFile")
            .arg("-")
            .arg(self.inner.as_os_str())
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

    fn list(&self) -> Result<Vec<PathBuf>> {
        let ls_out = Command::new("hdfs")
            .arg("dfs")
            .arg("-ls")
            .arg(self.inner.as_os_str())
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

    fn remove(&self) -> Result<()> {
        let rm_out = Command::new("hdfs")
            .arg("dfs")
            .arg("-rm")
            .arg(self.inner.as_os_str())
            .output()?;
        if rm_out.status.success() {
            Ok(())
        } else {
            Err(anyhow!("hdfs rm error"))
        }
    }

    fn append(&self, line: String) -> Result<()> {
        self.create(true)?;

        let mut append_cmd = Command::new("hdfs")
            .arg("dfs")
            .arg("-appendToFile")
            .arg("-")
            .arg(self.inner.as_os_str())
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
