package storage

func (s *Storage) Acquire() {

    s.refs.Add(1)

}

func (s *Storage) Release() {

    refs := s.refs.Add(-1)

    if refs <= 0 {

        s.buffer = nil

    }

}

func (s *Storage) RefCount() int64 {

    return s.refs.Load()

}